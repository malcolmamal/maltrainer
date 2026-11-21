"""
The Visage-Image VAE (f16, 96 latent channels) from ComfyUI's single file (``vae/visage_vae.safetensors``).

The latent is the posterior mean (no KL, no sampling). ``latent_mean`` / ``latent_std`` are dataset statistics:
``encode`` returns the normalized latents the DiT trains on and ``decode`` takes them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from maltrainer.models.visage.loading import strip_container_prefix
from maltrainer.models.visage.vae_config import VAEConfig
from maltrainer.models.visage.vae_decoder import ViTDecoder
from maltrainer.models.visage.vae_encoder import WaveletEncoder

logger = logging.getLogger(__name__)


class VisageVAE(nn.Module):
    latent_mean: Tensor
    latent_std: Tensor

    def __init__(self, config: VAEConfig) -> None:
        super().__init__()
        self.config = config
        self.wavelet_encoder = WaveletEncoder(
            config.latent_channels, config.encoder_widths, config.encoder_attention, config.encoder_attention_f16
        )
        self.vit_decoder = ViTDecoder(config)
        self.register_buffer("latent_mean", torch.zeros(config.latent_channels))
        self.register_buffer("latent_std", torch.ones(config.latent_channels))

    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        """RGB [B, 3, H, W] in [0, 1] (H and W multiples of 16) -> normalized latents [B, C, H/16, W/16]."""
        raw = cast(Tensor, self.wavelet_encoder((pixels * 2 - 1).to(self.latent_mean.dtype)))
        return (raw - self.latent_mean[None, :, None, None]) / self.latent_std[None, :, None, None]

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Normalized latents [B, C, h, w] -> RGB [B, 3, 16h, 16w] in [0, 1]."""
        raw = latents.to(self.latent_mean.dtype) * self.latent_std[None, :, None, None]
        image = cast(Tensor, self.vit_decoder(raw + self.latent_mean[None, :, None, None]))
        return ((image.float() + 1) / 2).clamp(0, 1)


def load_visage_vae(
    path: str | Path, device: torch.device, dtype: torch.dtype = torch.float32, config: VAEConfig | None = None
) -> VisageVAE:
    """The frozen VAE on ``device``; fp32 by default (a bf16 file is exact in fp32, and it only runs to cache the
    latents and to render previews)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Visage-Image VAE not found at {path}")
    state = strip_container_prefix(load_file(str(path)))
    vae = VisageVAE(config or VAEConfig()).to(dtype)
    try:
        vae.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"{path.name} is not the Visage-Image VAE: {str(error).splitlines()[0]}") from error
    vae.requires_grad_(False)
    vae.to(device).eval()
    logger.info("Visage-Image VAE ready (%s, %s)", dtype, path.name)
    return vae
