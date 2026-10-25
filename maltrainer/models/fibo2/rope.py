"""Fibo 2.0 positions: a (plane, row, column) id per token, rotated pairwise with theta 256 per axis."""

from __future__ import annotations

import torch

from maltrainer.models.fibo2.config import ROPE_AXES, ROPE_THETA


def rope_angles(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, 3] positions -> (cos, sin), each [1, N, 1, 64] fp32; the phases are made in fp64, as diffusers does."""
    angles = []
    for axis, dim in enumerate(ROPE_AXES):
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float64, device=ids.device) / dim))
        angles.append(torch.outer(ids[:, axis].double(), freqs).float())
    joined = torch.cat(angles, dim=-1)[None, :, None]
    return joined.cos(), joined.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, N, H, D]: each adjacent (real, imaginary) pair rotated in fp32, rounded once to x's dtype."""
    real, imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).flatten(-2).type_as(x)


def grid_ids(plane: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """[height * width, 3] ids of one image on rope plane ``plane``."""
    rows, cols = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    return torch.stack((torch.full_like(rows, plane), rows, cols), dim=-1).flatten(0, 1)


def gist_ids(count: int, device: torch.device) -> torch.Tensor:
    """[count, 3] ids of the gist tokens: planes 1..count, row and column 0."""
    ids = torch.zeros(count, 3, dtype=torch.long, device=device)
    ids[:, 0] = torch.arange(1, count + 1, device=device)
    return ids
