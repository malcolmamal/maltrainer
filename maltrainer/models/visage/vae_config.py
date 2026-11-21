"""The Visage-Image VAE's configuration (one released VAE: f16, 96 channels, 256 px)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VAEConfig:
    latent_channels: int = 96
    encoder_widths: tuple[int, int, int, int] = (120, 240, 376, 496)
    encoder_attention: int = 2  # attention blocks after the f8 aggregator
    encoder_attention_f16: int = 2  # attention blocks after the f16 downsample
    decoder_width: int = 768
    decoder_depth: int = 6  # f16 stage
    decoder_heads: int = 12
    hires_width: int = 768
    hires_depth: int = 6  # f8 stage
    hires_heads: int = 12
    refiner_width: int = 48
    refiner2_width: int = 96
    layer_scale: float = 1e-4
