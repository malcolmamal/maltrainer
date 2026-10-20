"""Normalisation, convolution, resampling and attention blocks of the Qwen-Image-2.1 VAE."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm3d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = F.normalize(x.float(), dim=1, eps=self.eps).to(x.dtype)
        gamma = self.gamma.view(1, -1, 1, 1)
        return cast(torch.Tensor, norm * (self.gamma.shape[0] ** 0.5) * gamma)


class RMSNormAttn(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = F.normalize(x.float(), dim=1, eps=self.eps).to(x.dtype)
        gamma = self.gamma.view(1, -1, 1, 1)
        return cast(torch.Tensor, norm * (self.gamma.shape[0] ** 0.5) * gamma)


class CausalConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.squeeze(2)
        return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm3d(in_channels),
            nn.SiLU(),
            CausalConv3d(in_channels, out_channels, 3, padding=1),
            RMSNorm3d(out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            CausalConv3d(out_channels, out_channels, 3, padding=1),
        )
        if in_channels != out_channels:
            self.shortcut: nn.Module = CausalConv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.shortcut(x) + self.residual(x))


class Resample2D(nn.Module):
    def __init__(self, channels: int, mode: str = "down"):
        super().__init__()
        self.mode = mode
        if mode == "down":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(channels, channels, 3, stride=(2, 2), padding=0),
            )
        else:
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest"),
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.resample(x))


class Resample3D(nn.Module):
    def __init__(self, channels: int, mode: str = "down"):
        super().__init__()
        self.mode = mode
        if mode == "down":
            self.time_conv = nn.Conv3d(channels, channels, kernel_size=1)
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(channels, channels, 3, stride=(2, 2), padding=0),
            )
        else:
            self.time_conv = nn.Conv3d(channels, channels * 2, kernel_size=1)
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest"),
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.resample(x))


class VAEAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = RMSNormAttn(channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.to_qkv(norm_x).view(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).view(b, c, h, w)
        return cast(torch.Tensor, x + self.proj(attn))
