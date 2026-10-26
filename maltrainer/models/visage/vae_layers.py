"""The Visage-Image VAE encoder's building blocks: conv residual blocks, spatial attention, the Haar wavelet level and
the DC-AE fold."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def groups(channels: int) -> int:
    divisors = [g for g in (32, 16, 8, 4, 2, 1) if channels % g == 0]
    wide = [g for g in divisors if channels // g >= 8]
    return wide[0] if wide else divisors[-1]


class ConvResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        return cast(Tensor, self.conv2(F.silu(self.norm2(h))) + self.skip(x))


def res_blocks(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*(ConvResBlock(channels, channels) for _ in range(count)))


class SpatialAttention(nn.Module):
    """Residual multi-head self-attention over the pixels of a feature map."""

    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)
        y = self.attn(y, y, y, need_weights=False)[0]
        return cast(Tensor, x + self.proj(y).transpose(1, 2).reshape(b, c, h, w))


def attention_blocks(channels: int, count: int) -> nn.Sequential:
    return nn.Sequential(*(SpatialAttention(channels) for _ in range(count)))


class HaarDWT(nn.Module):
    """One orthonormal Haar level: [B, C, H, W] -> [B, 4C, H/2, W/2], LL band first."""

    kernel: Tensor

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        low, high = torch.tensor((1.0, 1.0)) / 2**0.5, torch.tensor((1.0, -1.0)) / 2**0.5
        bands = torch.stack([r[:, None] * c[None, :] for r, c in ((low, low), (low, high), (high, low), (high, high))])
        self.register_buffer(
            "kernel", bands[None].expand(channels, 4, 2, 2).reshape(channels * 4, 1, 2, 2).contiguous()
        )

    def forward(self, x: Tensor) -> Tensor:
        # fp32: detail bands are small differences that bf16 would quantize away.
        with torch.autocast(device_type=x.device.type, enabled=False):
            bands = F.conv2d(x.float(), self.kernel.float(), stride=2, groups=self.channels)
        bands = bands.to(self.kernel.dtype)
        b, _, h, w = bands.shape
        return bands.view(b, self.channels, 4, h, w).transpose(1, 2).reshape(b, 4 * self.channels, h, w)


def fold_mean(x: Tensor, factor: int, out_channels: int) -> Tensor:
    """Pixel-unshuffle by ``factor``, then average channel groups down to ``out_channels``."""
    folded = F.pixel_unshuffle(x, factor)
    b, c, h, w = folded.shape
    return folded.view(b, out_channels, c // out_channels, h, w).mean(dim=2)


class DCAEDown(nn.Module):
    """2x downsample: conv + pixel-unshuffle, plus a parameter-free fold_mean shortcut."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels // 4, 3, padding=1)
        self.channels = channels

    def forward(self, x: Tensor) -> Tensor:
        return F.pixel_unshuffle(self.conv(x), 2) + fold_mean(x, 2, self.channels)
