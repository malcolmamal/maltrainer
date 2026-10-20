"""Embedding, attention and feed-forward layers of the MiniMax H3 transformer."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 0,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    if timesteps.dim() != 1:
        raise ValueError("Timesteps should be a 1d-tensor")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = scale * (timesteps[:, None].float() * torch.exp(exponent)[None, :])
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb


def _apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, passthrough), dim=-1).contiguous()


class MiniMaxH3RotaryPosEmbed(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, rope_freq_dim: int, rope_theta: float):
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        time, height, width = frequencies.unbind(dim=1)
        frequencies = torch.cat((time, height, width), dim=-1)
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies.cos(), frequencies.sin()


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(self, freq_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = get_timestep_embedding(
            timestep,
            self.freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        embedding = embedding.to(self.proj_in.weight.dtype)
        return cast(torch.Tensor, self.proj_out(F.silu(self.proj_in(embedding))))


class MiniMaxH3AdaLNProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, apply_silu: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.apply_silu = apply_silu

    def forward(self, timestep_embedding: torch.Tensor) -> torch.Tensor:
        compute_dtype = getattr(self.linear, "scale_weight", self.linear.weight).dtype
        activated = F.silu(timestep_embedding) if self.apply_silu else timestep_embedding
        activated = activated.to(compute_dtype)
        return cast(torch.Tensor, self.linear(activated))


class MiniMaxH3Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, head_dim: int, qk_norm_eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.qkv_proj = nn.Linear(hidden_size, 3 * self.inner_dim, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query, key, value = self.qkv_proj(hidden_states).chunk(3, dim=-1)
        query = self.q_norm(query.unflatten(-1, (self.heads, self.head_dim)))
        key = self.k_norm(key.unflatten(-1, (self.heads, self.head_dim)))
        value = value.unflatten(-1, (self.heads, self.head_dim))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask[None, None, :, :]
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(query.dtype)
        return cast(torch.Tensor, self.out_proj(hidden_states))


class MiniMaxH3FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 2 * ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc1(hidden_states).chunk(2, dim=-1)
        return cast(torch.Tensor, self.fc2(F.silu(gate) * value))
