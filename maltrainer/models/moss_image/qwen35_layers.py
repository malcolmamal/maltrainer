"""
Qwen3.5 text model building blocks: norms, MLP and gated full attention (see ``text_encoder``).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from maltrainer.models.moss_image.qwen35_config import Qwen35TextConfig


class RMSNorm(nn.Module):
    """``add=True`` scales by (1 + weight), Qwen3.5's convention for every norm but the gated one."""

    def __init__(self, dim: int, eps: float, add: bool):
        super().__init__()
        self.eps = eps
        self.add = add
        self.weight = nn.Parameter(torch.zeros(dim) if add else torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight + 1.0 if self.add else self.weight
        return F.rms_norm(x, weight.shape, weight.to(x.dtype), self.eps)


class RMSNormGated(RMSNorm):
    def __init__(self, dim: int, eps: float):
        super().__init__(dim, eps, add=False)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:  # type: ignore[override]  # gated variant: second arg by design
        return super().forward(x) * F.silu(gate.to(x.dtype))


class MLP(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class GatedAttention(nn.Module):
    def __init__(self, config: Qwen35TextConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = int(config.head_dim * config.partial_rotary_factor)
        inner = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, inner * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(inner, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps, add=True)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps, add=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        qg = self.q_proj(x).view(B, L, self.num_heads, self.head_dim * 2)
        q, gate = qg[..., : self.head_dim], qg[..., self.head_dim :].reshape(B, L, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Partial split-half RoPE on the first rotary_dim dims, in fp32, rounded once.
        r = self.rotary_dim
        q_rot, k_rot = q[..., :r].float(), k[..., :r].float()
        q = torch.cat([(q_rot * cos + rotate_half(q_rot) * sin).to(v.dtype), q[..., r:]], dim=-1)
        k = torch.cat([(k_rot * cos + rotate_half(k_rot) * sin).to(v.dtype), k[..., r:]], dim=-1)

        rep = self.num_heads // self.num_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return cast(torch.Tensor, self.o_proj(out * gate.sigmoid()))
