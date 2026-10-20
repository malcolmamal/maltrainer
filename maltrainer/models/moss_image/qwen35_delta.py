"""
Qwen3.5 Gated DeltaNet linear attention and the chunked gated delta rule (see ``text_encoder``).
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from maltrainer.models.moss_image.qwen35_config import Qwen35TextConfig
from maltrainer.models.moss_image.qwen35_layers import RMSNormGated


def chunk_gated_delta_rule(
    query: Tensor, key: Tensor, value: Tensor, g: Tensor, beta: Tensor, chunk_size: int = 64
) -> Tensor:
    """ComfyUI's torch_chunk_gated_delta_rule from a zero state. [B, L, H, D] in, [B, L, H, Dv] out."""
    initial_dtype = query.dtype
    query = F.normalize(query, dim=-1)
    key = F.normalize(key, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = [F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value)]
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ state
        core_attn_out[:, :, i] = (q_i * g[:, :, i, :, None].exp()) @ state + attn @ v_new
        state = (
            state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    return core_attn_out[:, :, :sequence_length].transpose(1, 2).contiguous().to(initial_dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        hidden = config.hidden_size
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.key_dim = self.num_key_heads * self.key_head_dim
        self.value_dim = self.num_value_heads * self.value_head_dim
        conv_dim = self.key_dim * 2 + self.value_dim

        self.in_proj_qkv = nn.Linear(hidden, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden, self.num_value_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, hidden, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_value_heads))
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=config.conv_kernel_size,
            groups=conv_dim,
            padding=config.conv_kernel_size - 1,
            bias=False,
        )
        self.norm = RMSNormGated(self.value_head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        z = self.in_proj_z(x)
        mixed = self.in_proj_qkv(x).transpose(1, 2)
        mixed = F.silu(
            F.conv1d(
                mixed, self.conv1d.weight.to(mixed.dtype), None, padding=self.conv1d.padding, groups=self.conv1d.groups
            )[:, :, :L]
        ).transpose(1, 2)
        query, key, value = cast(Any, mixed).split([self.key_dim, self.key_dim, self.value_dim], dim=-1)

        beta = self.in_proj_b(x).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias.float())
        query = query.reshape(B, L, self.num_key_heads, self.key_head_dim)
        key = key.reshape(B, L, self.num_key_heads, self.key_head_dim)
        value = value.reshape(B, L, self.num_value_heads, self.value_head_dim)
        if self.num_value_heads != self.num_key_heads:
            rep = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        core = chunk_gated_delta_rule(query, key, value, g=g, beta=beta)
        core = self.norm(core.reshape(-1, self.value_head_dim), z.reshape(-1, self.value_head_dim))
        return cast(torch.Tensor, self.out_proj(core.reshape(B, L, -1)))
