"""Supra2-IMG DiT building blocks (AdaLN-Zero blocks, timestep embedder, final layer)."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None] * 1000.0
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.mlp(self._sinusoidal(t)))


class Attention(nn.Module):
    """Multi-head self- or cross-attention."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: int | None = None) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.is_self = ctx_dim is None
        if ctx_dim is None:
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
        else:
            self.q = nn.Linear(dim, dim, bias=True)
            self.kv = nn.Linear(ctx_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(
        self, x: torch.Tensor, ctx: torch.Tensor | None = None, ctx_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, length, channels = x.shape
        if self.is_self:
            qkv = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        else:
            ctx_len = cast(torch.Tensor, ctx).shape[1]
            q = self.q(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            kv = self.kv(ctx).view(batch, ctx_len, 2, self.n_heads, self.head_dim)
            k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        attn_mask = None
        if ctx_mask is not None:
            attn_mask = ctx_mask.bool()[:, None, None, :]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch, length, channels)
        return cast(torch.Tensor, self.proj(out))


class DiTBlock(nn.Module):
    """DiT block: AdaLN-Zero self-attn + cross-attn + MLP."""

    def __init__(self, dim: int, n_heads: int, ctx_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, n_heads)
        self.norm_ca = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, n_heads, ctx_dim=ctx_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, dim),
        )
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        ctx: torch.Tensor,
        ctx_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = self.adaln(c).chunk(6, dim=1)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_sa, scale_sa))
        x = x + self.cross_attn(self.norm_ca(x), ctx=ctx, ctx_mask=ctx_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_ch, bias=True)
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln(c).chunk(2, dim=1)
        return cast(torch.Tensor, self.linear(modulate(self.norm(x), shift, scale)))
