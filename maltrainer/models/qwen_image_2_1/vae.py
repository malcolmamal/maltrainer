"""
Qwen-Image-2.1 3D VAE Encoder and Decoder.
Supports 16x spatial downsampling, 64 latent channels, and per-channel mean/std normalization.

The layers live in ``vae_blocks``, ``vae_stages`` and ``vae_networks``, the normalisation constants in
``vae_constants``; every name is re-exported here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import safetensors.torch
import torch
from torch import nn

from maltrainer.models.qwen_image_2_1.vae_blocks import (
    CausalConv3d,
    Resample2D,
    Resample3D,
    ResBlock,
    RMSNorm3d,
    RMSNormAttn,
    VAEAttention,
)
from maltrainer.models.qwen_image_2_1.vae_constants import LATENTS_MEAN, LATENTS_STD
from maltrainer.models.qwen_image_2_1.vae_networks import (
    Qwen21VAEDecoder,
    Qwen21VAEEncoder,
)
from maltrainer.models.qwen_image_2_1.vae_stages import (
    AvgDown3D,
    DecoderStage,
    DupUp3D,
    EncoderStage,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LATENTS_MEAN",
    "LATENTS_STD",
    "AvgDown3D",
    "CausalConv3d",
    "DecoderStage",
    "DupUp3D",
    "EncoderStage",
    "Qwen21VAEDecoder",
    "Qwen21VAEEncoder",
    "QwenImage21VAE",
    "RMSNorm3d",
    "RMSNormAttn",
    "ResBlock",
    "Resample2D",
    "Resample3D",
    "VAEAttention",
    "load_qwen_image_2_1_vae",
]


class QwenImage21VAE(nn.Module):
    """Qwen-Image-2.1 Complete VAE Wrapper."""

    def __init__(self, include_decoder: bool = False, z_dim: int = 64):
        super().__init__()
        self.encoder = Qwen21VAEEncoder(z_dim=z_dim)
        self.decoder = Qwen21VAEDecoder(z_dim=z_dim) if include_decoder else None

        # The checkpoint keeps these at its root as `conv1` / `conv2`: the encoder head's
        # moments go through quant_conv before the mean/logvar split, and latents go through
        # post_quant_conv before the decoder. Skipping them puts the latents in a different
        # space than the one the transformer was trained on.
        self.quant_conv = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = CausalConv3d(z_dim, z_dim, 1) if include_decoder else None

        self.register_buffer(
            "latents_mean", torch.tensor(LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1), persistent=False
        )
        self.register_buffer(
            "latents_std", torch.tensor(LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1), persistent=False
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize latents using channel-wise mean and std."""
        mean = self.get_buffer("latents_mean").to(device=latents.device, dtype=latents.dtype)
        std = self.get_buffer("latents_std").to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) / std

    def unnormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Unnormalize latents using channel-wise mean and std."""
        mean = self.get_buffer("latents_mean").to(device=latents.device, dtype=latents.dtype)
        std = self.get_buffer("latents_std").to(device=latents.device, dtype=latents.dtype)
        return latents * std + mean

    def encode(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Encode images [B, 3, H, W] into normalized latents [B, 64, H/16, W/16].

        Input images should be in range [-1.0, 1.0] or [0.0, 1.0].
        Note: If in [0, 1], input is scaled to [-1, 1] internally.
        """
        if x.min() >= 0.0 and x.max() <= 1.0:
            x = x * 2.0 - 1.0

        # VAE conv1 expects 4 input channels [B, 4, H, W] (RGBA in [-1, 1], opaque alpha = 1.0)
        if x.shape[1] == 3:
            pad = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        moments = self.quant_conv(self.encoder(x))
        mean, logvar = moments.chunk(2, dim=1)

        if deterministic:
            latents = mean
        else:
            std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))
            latents = mean + std * torch.randn_like(mean)

        return self.normalize_latents(latents)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode normalized latents [B, 64, H/16, W/16] into images [B, 3, H, W] in range [0, 1]."""
        if self.decoder is None or self.post_quant_conv is None:
            raise RuntimeError("Decoder was not loaded for this VAE instance.")
        z = self.unnormalize_latents(z)
        out = self.decoder(self.post_quant_conv(z))
        # Take first 3 channels (RGB)
        rgb = out[:, :3]
        # Rescale from [-1, 1] to [0, 1]
        rgb = ((rgb + 1.0) / 2.0).clamp(0.0, 1.0)
        return cast(torch.Tensor, rgb)


def load_qwen_image_2_1_vae(
    vae_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    include_decoder: bool = False,
) -> QwenImage21VAE:
    """Load Qwen-Image-2.1 VAE from safetensors checkpoint."""
    vae_path = Path(vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")

    logger.info("Loading Qwen-Image-2.1 VAE from %s...", vae_path)
    model = QwenImage21VAE(include_decoder=include_decoder)

    with safetensors.safe_open(str(vae_path), framework="pt") as f:
        # Load encoder weights
        enc_sd = {}
        for k in f.keys():
            if k.startswith("encoder."):
                enc_sd[k[len("encoder.") :]] = f.get_tensor(k)
        model.encoder.load_state_dict(enc_sd, strict=True)

        # Root-level `conv1` / `conv2` are quant_conv / post_quant_conv, not encoder layers.
        quant_sd = {k[len("conv1.") :]: f.get_tensor(k) for k in f.keys() if k.startswith("conv1.")}
        if not quant_sd:
            raise KeyError(f"VAE checkpoint {vae_path} has no root `conv1.*` (quant_conv) weights")
        model.quant_conv.load_state_dict(quant_sd, strict=True)

        if include_decoder:
            dec_sd = {}
            for k in f.keys():
                if k.startswith("decoder."):
                    dec_sd[k[len("decoder.") :]] = f.get_tensor(k)
            cast(Qwen21VAEDecoder, model.decoder).load_state_dict(dec_sd, strict=True)

            post_sd = {k[len("conv2.") :]: f.get_tensor(k) for k in f.keys() if k.startswith("conv2.")}
            if not post_sd:
                raise KeyError(f"VAE checkpoint {vae_path} has no root `conv2.*` (post_quant_conv) weights")
            cast(CausalConv3d, model.post_quant_conv).load_state_dict(post_sd, strict=True)

    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    model.eval()
    logger.info("Loaded Qwen-Image-2.1 VAE successfully on %s in %s", device, dtype)
    return model
