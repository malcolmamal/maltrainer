"""The DAC-lineage waveform encoder layers of the MiniMax H3 audio VAE."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn


def _snake(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # the snake activation: x plus sin squared of (alpha * x), scaled by 1 / beta
    return x + (beta + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(x.dtype)
        return _snake(x, alpha, alpha)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return cast(torch.Tensor, y + x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(x))


class DacEncoder(nn.Module):
    def __init__(self, d_model: int = 64, strides: tuple[int, ...] = (2, 4, 4, 5, 5), d_latent: int = 2048):
        super().__init__()
        block: list[nn.Module] = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block.append(EncoderBlock(d_model, stride=stride))
        block += [Snake1d(d_model), nn.Conv1d(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.block(x))
