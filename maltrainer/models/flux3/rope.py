"""FLUX.3 position and timestep maths: sinusoidal timestep embedding, 4-axis RoPE and the token ids."""

from __future__ import annotations

import math

import torch

IMAGE_ROPE_SCALE = 0.5
ROPE_THETA = 10000
ROPE_AXES = (32, 32, 32, 32)


def timestep_embedding(
    t: torch.Tensor, dim: int = 256, max_period: int = 10000, time_factor: float = 1000.0
) -> torch.Tensor:
    """comfy/ldm/flux/layers.py: the x1000 happens in ``t``'s own dtype."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding.to(t) if torch.is_floating_point(t) else embedding


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """[..., n] positions -> [..., n, dim/2, 2, 2] rotation matrices (comfy/ldm/flux/math.py)."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float32).to(torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def rope_from_ids(ids: torch.Tensor) -> torch.Tensor:
    """[N, 4] ids -> [1, 1, N, 64, 2, 2], broadcast over batch and heads of [B, H, N, D]."""
    emb = torch.cat([rope(ids[..., i], ROPE_AXES[i], ROPE_THETA) for i in range(ids.shape[-1])], dim=-3)
    return emb[None, None]


def apply_rope(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    x_ = x.to(pe.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    out = pe[..., 0] * x_[..., 0] + pe[..., 1] * x_[..., 1]
    return out.reshape(*x.shape).type_as(x)


def image_ids(h: int, w: int, index: int, device: torch.device | str) -> torch.Tensor:
    ids = torch.zeros(h, w, 4, device=device, dtype=torch.float32)
    ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None] * IMAGE_ROPE_SCALE
    ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :] * IMAGE_ROPE_SCALE
    ids[..., 3] = index
    return ids.flatten(0, 1)


def text_ids(length: int, device: torch.device | str) -> torch.Tensor:
    ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, device=device, dtype=torch.float32)
    return ids
