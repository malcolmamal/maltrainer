"""Krea 2 positions and timesteps: the sinusoidal embedding and Flux-style RoPE (comfy/ldm/krea2/model.py)."""

from __future__ import annotations

import math

import torch


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    """Flux's sinusoidal embedding (cos first), t scaled by 1000."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rope(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """Flux RoPE rotation matrices for one axis: [B, L, dim/2, 2, 2] fp32."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def embed_nd(ids: torch.Tensor, axes_dim: list[int] | tuple[int, ...], theta: float) -> torch.Tensor:
    emb = torch.cat([rope(ids[..., i], axes_dim[i], theta) for i in range(ids.shape[-1])], dim=-3)
    return emb.unsqueeze(1)  # [B, 1, L, headdim/2, 2, 2]


def apply_rope1(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * x_[..., 0] + freqs[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).type_as(x)
