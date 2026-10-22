"""The Flux2 VAE's convolutional encoder and decoder (SD-style AutoencoderKL layout, the file's own key names)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = _norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if hasattr(self, "nin_shortcut"):
            x = self.nin_shortcut(x)
        return cast(torch.Tensor, x + h)


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        B, C, H, W = h.shape
        q, k, v = (t.view(B, 1, C, H * W).transpose(2, 3) for t in (self.q(h), self.k(h), self.v(h)))
        out = F.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(B, C, H, W)
        return cast(torch.Tensor, x + self.proj_out(out))


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest")))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(F.pad(x, (0, 1, 0, 1))))


class Encoder(nn.Module):
    def __init__(
        self, ch: int = 128, ch_mult: Sequence[int] = (1, 2, 4, 4), num_res_blocks: int = 2, z_channels: int = 32
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        in_mult = (1, *tuple(ch_mult))
        self.down: Any = nn.ModuleList()
        block_in = ch
        for level, mult in enumerate(ch_mult):
            down = nn.Module()
            block_in, block_out = ch * in_mult[level], ch * mult
            down.block = nn.ModuleList()
            for _ in range(num_res_blocks):
                down.block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            if level != len(ch_mult) - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid: Any = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = _norm(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for down in self.down:
            for block in down.block:
                h = block(h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return cast(torch.Tensor, self.conv_out(F.silu(self.norm_out(h))))


class Decoder(nn.Module):
    def __init__(
        self, ch: int = 128, ch_mult: Sequence[int] = (1, 2, 4, 4), num_res_blocks: int = 2, z_channels: int = 32
    ):
        super().__init__()
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, padding=1)
        self.mid: Any = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        # up[0] is the full-resolution level; it runs last
        self.up: Any = nn.ModuleList()
        for level in reversed(range(len(ch_mult))):
            up = nn.Module()
            up.block = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                up.block.append(ResnetBlock(block_in, ch * ch_mult[level]))
                block_in = ch * ch_mult[level]
            if level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = _norm(block_in)
        self.conv_out = nn.Conv2d(block_in, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        for level in reversed(range(len(self.up))):
            for block in self.up[level].block:
                h = block(h)
            if hasattr(self.up[level], "upsample"):
                h = self.up[level].upsample(h)
        return cast(torch.Tensor, self.conv_out(F.silu(self.norm_out(h))))
