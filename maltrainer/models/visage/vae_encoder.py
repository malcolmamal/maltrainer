"""The Visage-Image VAE encoder (LiteVAE style): a 3-level Haar pyramid, one conv extractor per level, aggregation on
the f8 grid, a DC-AE downsample to f16, and two parameter-free pixel shortcuts. The latent is the posterior mean."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from maltrainer.models.visage.vae_layers import DCAEDown, HaarDWT, attention_blocks, fold_mean, groups, res_blocks


class WaveletEncoder(nn.Module):
    def __init__(
        self, latent_channels: int, widths: tuple[int, int, int, int], attention: int = 2, attention_f16: int = 2
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.dwt = HaarDWT(3)
        self.extractors = nn.ModuleList()
        self.reducers = nn.ModuleList()
        for level, width in enumerate(widths[:3], start=1):
            self.extractors.append(nn.Sequential(nn.Conv2d(12, width, 3, padding=1), res_blocks(width, 2)))
            stride = 2 ** (3 - level)  # bring level `level` down to the f8 grid
            self.reducers.append(nn.AvgPool2d(stride) if stride > 1 else nn.Identity())
        agg = widths[3]
        self.aggregate = nn.Sequential(
            nn.Conv2d(sum(widths[:3]), agg, 1), res_blocks(agg, 4), attention_blocks(agg, attention)
        )
        self.down16 = nn.Sequential(DCAEDown(agg), res_blocks(agg, 2), attention_blocks(agg, attention_f16))
        self.global_skip = nn.Conv2d(3 * 16**2, agg, 1)
        self.to_moments = nn.Sequential(
            nn.GroupNorm(groups(agg), agg), nn.SiLU(), nn.Conv2d(agg, 2 * latent_channels, 1)
        )

    def forward(self, image: Tensor) -> Tensor:
        """[-1, 1] images [B, 3, H, W] (H and W multiples of 16) -> raw latents [B, C, H/16, W/16]."""
        features, low = [], image
        for extractor, reducer in zip(self.extractors, self.reducers, strict=True):
            bands = self.dwt(low)
            low = bands[:, :3]
            features.append(reducer(extractor(bands)))
        h = self.down16(self.aggregate(torch.cat(features, dim=1)))
        h = h + self.global_skip(F.pixel_unshuffle(image, 16))
        mean = self.to_moments(h)[:, : self.latent_channels]  # the moments' mean half: the latent is deterministic
        return cast(Tensor, mean + fold_mean(image, 16, self.latent_channels))
