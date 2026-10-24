"""Fibo 2.0 DiT layers; the module names match ComfyUI's single file exactly, so it loads with ``strict=True``."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from maltrainer.models.fibo2.config import ADALN_DIM
from maltrainer.models.fibo2.rope import apply_rope


class RMSNorm(nn.Module):
    """Normalises in fp32, rounds to the input dtype, then scales (diffusers' RMSNorm)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps).to(x.dtype) * self.weight.to(x.dtype)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[B, N, H, D] in, [B, N, H * D] out."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype)).transpose(1, 2).flatten(2)


class JointAttention(nn.Module):
    """Self-attention over the whole sequence, with RoPE and QK-norm."""

    def __init__(self, dim: int, heads: int, eps: float):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(dim // heads, eps)
        self.k_norm = RMSNorm(dim // heads, eps)
        self.heads = heads

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).unflatten(-1, (3, self.heads, -1)).unbind(2)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        return cast(torch.Tensor, self.out(attend(q, k, v)))


class Attention(nn.Module):
    """Attention to a context of width ``kv_dim``, without RoPE: the text cross-attention and the Perceiver's."""

    def __init__(self, dim: int, heads: int, kv_dim: int, eps: float):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.norm_q = RMSNorm(dim // heads, eps)
        self.norm_k = RMSNorm(dim // heads, eps)
        self.heads = heads

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        k, v = self.to_kv(context).unflatten(-1, (2, self.heads, -1)).unbind(2)
        return cast(torch.Tensor, self.to_out(attend(q, self.norm_k(k), v)))


class FeedForward(nn.Module):
    """SwiGLU with w1 and w3 fused into ``w13`` (ComfyUI's layout: gate first, then up)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w13 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).chunk(2, dim=-1)
        return cast(torch.Tensor, self.w2(F.silu(gate) * up))


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of t in [0, 1000], then an MLP down to the 256-wide adaLN input."""

    def __init__(self, freq_dim: int = ADALN_DIM, hidden: int = 1024):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, ADALN_DIM))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat((args.cos(), args.sin()), dim=-1)
        return cast(torch.Tensor, self.mlp(embedding.to(next(self.mlp.parameters()).dtype)))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(ADALN_DIM, dim))

    def forward(self, x: torch.Tensor, adaln: torch.Tensor) -> torch.Tensor:
        scale = 1 + self.adaLN_modulation(adaln)
        return cast(torch.Tensor, self.linear(F.layer_norm(x, (x.shape[-1],), eps=1e-6) * scale[:, None]))
