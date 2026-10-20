"""
MOSS-Image DiT: the sinusoidal timestep embedding, 4-axis RoPE and the SiTU activation (see ``model``).
"""

from __future__ import annotations

from typing import Any, cast

import torch

ROPE_THETA = 10000
ROPE_AXES = 4
LATENT_CHANNELS = 128
TIME_FREQ_DIM = 256
TIME_FACTOR = 1000.0


def timestep_embedding(sigma: torch.Tensor, dim: int = TIME_FREQ_DIM, max_period: int = 10000) -> torch.Tensor:
    """[B] sigma -> [B, dim] float32 (cos | sin) of 1000 * sigma, the phases in float64."""
    half = dim // 2
    freqs = max_period ** (-torch.arange(half, device=sigma.device, dtype=torch.float64) / half)
    phase = (TIME_FACTOR * sigma.double())[:, None] * freqs
    return torch.cat([phase.cos(), phase.sin()], dim=-1).float()


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """[n] positions -> [n, dim/2, 2, 2] rotation matrices (comfy/ldm/flux/math.py)."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float32).to(torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def position_ids(h: int, w: int, text_len: int, device: torch.device) -> torch.Tensor:
    """[h*w + text_len, 4] (r, f, h, w): the image first, at f = text_len, rows/cols centred; then text."""
    ids = torch.zeros(h * w + text_len, 4, device=device, dtype=torch.float32)
    ids[: h * w, 1] = text_len
    ids[: h * w, 2] = (torch.arange(h, device=device, dtype=torch.float32) - (h - 1) / 2).repeat_interleave(w)
    ids[: h * w, 3] = (torch.arange(w, device=device, dtype=torch.float32) - (w - 1) / 2).repeat(h)
    ids[h * w :, 1] = torch.arange(text_len, device=device, dtype=torch.float32)
    return ids


def rope_from_ids(ids: torch.Tensor, head_dim: int) -> torch.Tensor:
    """[N, 4] ids -> [1, N, 1, head_dim/2, 2, 2], broadcast over batch and heads of [B, N, H, D]."""
    axis = head_dim // ROPE_AXES
    emb = torch.cat([rope(ids[:, i], axis, ROPE_THETA) for i in range(ROPE_AXES)], dim=-3)
    return emb[None, :, None]


def apply_rope(x: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Split-half RoPE on [B, N, H, D]: pairs (x[i], x[i + D/2]) (ComfyUI's rope_split_half)."""
    x_ = cast(Any, x.to(pe.dtype)).unflatten(-1, (2, -1)).movedim(-2, -1).unsqueeze(-2)
    out = pe[..., 0] * x_[..., 0] + pe[..., 1] * x_[..., 1]
    return cast(torch.Tensor, out.movedim(-1, -2).flatten(-2).type_as(x))


def situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))
