"""LTX-2.5 DiT layers; the module names match ComfyUI's single file exactly, so it loads with ``strict=True``."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from maltrainer.models.ltx25.rope import Rope, apply_rope

ADALN_EMBED = 256


def plain_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalisation without a weight (ComfyUI's ``rms_norm``)."""
    return F.rms_norm(x, (x.shape[-1],), eps=eps)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


class CrossAttention(nn.Module):
    """Attention with QK-norm across the heads, optional RoPE on queries and keys, and optional per-head gates."""

    def __init__(self, query_dim: int, context_dim: int | None, heads: int, head_dim: int, gated: bool):
        super().__init__()
        inner = heads * head_dim
        context_dim = query_dim if context_dim is None else context_dim
        self.heads, self.head_dim = heads, head_dim
        self.q_norm, self.k_norm = RMSNorm(inner, 1e-5), RMSNorm(inner, 1e-5)
        self.to_q = nn.Linear(query_dim, inner)
        self.to_k = nn.Linear(context_dim, inner)
        self.to_v = nn.Linear(context_dim, inner)
        self.to_gate_logits = nn.Linear(query_dim, heads) if gated else None
        self.to_out = nn.Sequential(nn.Linear(inner, query_dim), nn.Dropout(0.0))

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None, pe: Rope | None = None, k_pe: Rope | None = None
    ) -> torch.Tensor:
        context = x if context is None else context
        q, k, v = self.q_norm(self.to_q(x)), self.k_norm(self.to_k(context)), self.to_v(context)
        if pe is not None:
            q, k = apply_rope(q, pe), apply_rope(k, pe if k_pe is None else k_pe)
        batch, tokens = q.shape[:2]
        heads = (t.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(*heads).transpose(1, 2)
        if self.to_gate_logits is not None:
            out = out * (2.0 * torch.sigmoid(self.to_gate_logits(x))).unsqueeze(-1)
        return cast(torch.Tensor, self.to_out(out.reshape(batch, tokens, -1)))


class GELUApprox(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim: int, bias: bool, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            GELUApprox(dim, dim * mult, bias), nn.Dropout(0.0), nn.Linear(dim * mult, dim, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


def timestep_embedding(timesteps: torch.Tensor, dim: int = ADALN_EMBED, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding, cos half first (PixArt's ``flip_sin_to_cos``), frequencies from 1 to 1 / max_period."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    args = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat((torch.cos(args), torch.sin(args)), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.linear_2(self.act(self.linear_1(x))))


class TimestepEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(ADALN_EMBED, dim)

    def forward(self, timesteps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return cast(torch.Tensor, self.timestep_embedder(timestep_embedding(timesteps).to(dtype)))


class AdaLayerNormSingle(nn.Module):
    """A timestep -> ``coefficient * dim`` modulation vector, and the embedding it came from."""

    def __init__(self, dim: int, coefficient: int = 6):
        super().__init__()
        self.emb = TimestepEmbeddings(dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, coefficient * dim)

    def forward(self, timesteps: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.emb(timesteps, dtype)
        return cast(torch.Tensor, self.linear(self.silu(embedded))), embedded


def ada_values(table: torch.Tensor, timestep: torch.Tensor, rows: slice = slice(None)) -> tuple[torch.Tensor, ...]:
    """``table[rows] + timestep`` split per row: timestep is [B, T, n_rows * dim], the values are [B, T, dim] each."""
    batch, tokens = timestep.shape[:2]
    chosen = timestep.reshape(batch, tokens, table.shape[0], -1)[:, :, rows]
    return (table[rows][None, None].to(timestep.dtype) + chosen).unbind(dim=2)
