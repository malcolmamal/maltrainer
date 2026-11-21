"""
LTX-2.5 positions: fractional-position RoPE over (time, row, column) for video, time for audio and the 1D connector.

Mirrors comfy/ldm/lightricks (model.py ``generate_freqs`` / ``freqs_cis_matrix``, symmetric_patchifier.py): a token's
coordinates are scaled by the VAE strides, divided by the axis' maximum, and turned into ``cos`` / ``sin`` over a
geometric frequency grid. In split mode the two halves of each head rotate against each other, in interleaved mode
adjacent pairs do.
"""

from __future__ import annotations

import math

import torch

AUDIO_SAMPLE_RATE, AUDIO_HOP, AUDIO_DOWNSAMPLE = 16000, 160, 4

Rope = tuple[torch.Tensor, torch.Tensor, bool]  # cos, sin: [B, T, heads, head_dim / 2]; split flag


def frequency_grid(theta: float, n_pos: int, dim: int, double: bool, device: torch.device) -> torch.Tensor:
    """``dim // (2 n_pos)`` frequencies theta ** (0..1) * pi / 2, in fp64 when ``double``, rounded to fp32."""
    dtype = torch.float64 if double else torch.float32
    exponents = torch.linspace(0, 1, dim // (2 * n_pos), dtype=dtype, device=device)
    return (theta**exponents * math.pi / 2).float()


def make_rope(
    grid: torch.Tensor,
    dim: int,
    heads: int,
    max_pos: tuple[int, ...],
    theta: float,
    split: bool,
    double: bool,
    middle: bool,
) -> Rope:
    """grid [B, n_pos, T] or [B, n_pos, T, 2] (start, end) positions -> the rotation for ``heads`` heads of ``dim``."""
    if middle and grid.ndim == 4:
        grid = (grid[..., 0] + grid[..., 1]) / 2.0
    elif grid.ndim == 4:
        grid = grid[..., 0]
    n_pos = grid.shape[1]
    fractional = torch.stack([grid[:, i] / max_pos[i] for i in range(n_pos)], dim=-1)
    indices = frequency_grid(theta, n_pos, dim, double, grid.device)
    freqs = (indices * (fractional.unsqueeze(-1) * 2 - 1)).transpose(-1, -2).flatten(2)
    cos, sin = freqs.cos(), freqs.sin()
    pad = (dim // 2 - freqs.shape[-1]) if split else (dim % (2 * n_pos)) // 2
    if pad:
        cos = torch.cat((torch.ones_like(cos[..., :pad]), cos), dim=-1)
        sin = torch.cat((torch.zeros_like(sin[..., :pad]), sin), dim=-1)
    batch, tokens, half = cos.shape
    return cos.reshape(batch, tokens, heads, half // heads), sin.reshape(batch, tokens, heads, half // heads), split


def apply_rope(x: torch.Tensor, rope: Rope) -> torch.Tensor:
    """x [B, T, heads * head_dim] rotated in fp32 and rounded once to x's dtype."""
    cos, sin, split = rope
    batch = x.shape[0]
    _, tokens, heads, half = cos.shape
    pairs = (
        x.float().reshape(batch, tokens, heads, 2, half) if split else x.float().reshape(batch, tokens, heads, half, 2)
    )
    first, second = pairs.unbind(3 if split else 4)
    rotated = (first * cos - second * sin, first * sin + second * cos)
    return torch.stack(rotated, dim=3 if split else 4).reshape(x.shape).to(x.dtype)


def video_grid(frames: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    """[1, 3, F * H * W, 2] latent (start, end) coordinates of one latent patch per token."""
    axes = [torch.arange(n, dtype=torch.float32, device=device) for n in (frames, height, width)]
    start = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0).flatten(1)
    return torch.stack((start, start + 1), dim=-1).unsqueeze(0)


def video_positions(
    frames: int, height: int, width: int, scale: tuple[int, int, int], causal: bool, fps: float, device: torch.device
) -> torch.Tensor:
    """The video grid in pixel space (time in seconds), the coordinates RoPE and the cross-modal attention use."""
    pixel = video_grid(frames, height, width, device) * torch.tensor(scale, dtype=torch.float32, device=device).view(
        1, 3, 1, 1
    )
    if causal:
        pixel[:, 0] = (pixel[:, 0] + 1 - scale[0]).clamp(min=0)
    pixel[:, 0] = pixel[:, 0] * (1.0 / fps)
    return pixel


def audio_positions(frames: int, device: torch.device, causal: bool = True) -> torch.Tensor:
    """[1, 1, T, 2] start and end of each audio latent frame in seconds."""

    def seconds(index: torch.Tensor) -> torch.Tensor:
        mel = index * AUDIO_DOWNSAMPLE
        if causal:
            mel = (mel + 1 - AUDIO_DOWNSAMPLE).clip(min=0)
        return mel * AUDIO_HOP / AUDIO_SAMPLE_RATE

    index = torch.arange(frames, dtype=torch.float32, device=device)
    return torch.stack((seconds(index), seconds(index + 1)), dim=-1).unsqueeze(0).unsqueeze(1)
