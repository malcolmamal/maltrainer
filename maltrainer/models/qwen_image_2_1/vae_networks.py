"""The Qwen-Image-2.1 VAE encoder and decoder networks."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from maltrainer.models.qwen_image_2_1.vae_blocks import (
    CausalConv3d,
    Resample2D,
    Resample3D,
    ResBlock,
    RMSNorm3d,
    VAEAttention,
)
from maltrainer.models.qwen_image_2_1.vae_stages import AvgDown3D, DecoderStage, DupUp3D, EncoderStage


class Qwen21VAEEncoder(nn.Module):
    def __init__(self, base_dim: int = 96, z_dim: int = 64):
        super().__init__()
        self.conv1 = CausalConv3d(4, base_dim, 3, padding=1)
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8, base_dim * 8]
        # temperal_downsample = [False, True, True, True]; the last stage does not downsample
        self.downsamples = nn.ModuleList(
            [
                EncoderStage(
                    [ResBlock(dims[0], dims[0]), ResBlock(dims[0], dims[0])],
                    Resample2D(dims[0], "down"),
                    AvgDown3D(dims[0], dims[0], factor_t=1, factor_s=2),
                ),
                EncoderStage(
                    [ResBlock(dims[0], dims[1]), ResBlock(dims[1], dims[1])],
                    Resample3D(dims[1], "down"),
                    AvgDown3D(dims[0], dims[1], factor_t=2, factor_s=2),
                ),
                EncoderStage(
                    [ResBlock(dims[1], dims[2]), ResBlock(dims[2], dims[2])],
                    Resample3D(dims[2], "down"),
                    AvgDown3D(dims[1], dims[2], factor_t=2, factor_s=2),
                ),
                EncoderStage(
                    [ResBlock(dims[2], dims[3]), ResBlock(dims[3], dims[3])],
                    Resample3D(dims[3], "down"),
                    AvgDown3D(dims[2], dims[3], factor_t=2, factor_s=2),
                ),
                EncoderStage(
                    [ResBlock(dims[3], dims[4]), ResBlock(dims[4], dims[4])],
                    None,
                    AvgDown3D(dims[3], dims[4], factor_t=1, factor_s=1),
                ),
            ]
        )
        self.middle = nn.ModuleList(
            [
                ResBlock(dims[4], dims[4]),
                VAEAttention(dims[4]),
                ResBlock(dims[4], dims[4]),
            ]
        )
        self.head = nn.Sequential(
            RMSNorm3d(dims[4]),
            nn.SiLU(),
            CausalConv3d(dims[4], z_dim * 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for stage in self.downsamples:
            x = stage(x)
        for m in self.middle:
            x = m(x)
        x = self.head(x)
        return x


class Qwen21VAEDecoder(nn.Module):
    def __init__(self, z_dim: int = 64):
        super().__init__()
        self.conv1 = CausalConv3d(z_dim, 1152, 3, padding=1)
        self.middle = nn.ModuleList(
            [
                ResBlock(1152, 1152),
                VAEAttention(1152),
                ResBlock(1152, 1152),
            ]
        )
        # temperal_upsample = [True, True, True, False]; the last stage does not upsample
        self.upsamples = nn.ModuleList(
            [
                DecoderStage(
                    [ResBlock(1152, 1152), ResBlock(1152, 1152), ResBlock(1152, 1152)],
                    Resample3D(1152, "up"),
                    DupUp3D(1152, 1152, factor_t=2, factor_s=2),
                ),
                DecoderStage(
                    [ResBlock(1152, 1152), ResBlock(1152, 1152), ResBlock(1152, 1152)],
                    Resample3D(1152, "up"),
                    DupUp3D(1152, 1152, factor_t=2, factor_s=2),
                ),
                DecoderStage(
                    [ResBlock(1152, 576), ResBlock(576, 576), ResBlock(576, 576)],
                    Resample3D(576, "up"),
                    DupUp3D(1152, 576, factor_t=2, factor_s=2),
                ),
                DecoderStage(
                    [ResBlock(576, 288), ResBlock(288, 288), ResBlock(288, 288)],
                    Resample2D(288, "up"),
                    DupUp3D(576, 288, factor_t=1, factor_s=2),
                ),
                DecoderStage([ResBlock(288, 144), ResBlock(144, 144), ResBlock(144, 144)], None, None),
            ]
        )
        self.head = nn.Sequential(
            RMSNorm3d(144),
            nn.SiLU(),
            CausalConv3d(144, 4, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv1(z)
        for m in self.middle:
            x = m(x)
        for stage in self.upsamples:
            x = stage(x)
        x = self.head(x)
        return cast(torch.Tensor, x)
