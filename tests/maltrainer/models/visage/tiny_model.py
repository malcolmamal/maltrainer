"""Visage-Image components small enough for CPU tests, with weights from a name-seeded fill (so the same fill gives
the upstream package's modules the same weights, which is how the golden values in test_golden.py were made)."""

from __future__ import annotations

import zlib

import torch
from torch import nn

from maltrainer.models.visage.config import VisageConfig
from maltrainer.models.visage.model import VisageTransformer
from maltrainer.models.visage.vae import VisageVAE
from maltrainer.models.visage.vae_config import VAEConfig

CAP = 16
LATENT = 8
TINY = VisageConfig(
    in_channels=LATENT,
    width=128,
    depth=2,
    mlp_ratio=2,
    text_dim=CAP,
    noise_embed_dims=16,
    text_length_buckets=(4, 8),
    n_text_registers=2,
)
TINY_VAE = VAEConfig(
    latent_channels=LATENT,
    encoder_widths=(16, 16, 16, 32),
    encoder_attention=1,
    encoder_attention_f16=1,
    decoder_width=64,
    decoder_depth=1,
    decoder_heads=2,
    hires_width=64,
    hires_depth=1,
    hires_heads=2,
    refiner_width=4,
    refiner2_width=4,
)
FIXED = ("angular_speeds", "kernel")  # buffers computed in __init__, not weights


def fill(model: nn.Module, scale: float = 0.2) -> None:
    """Every weight from a generator seeded by its name: norm scales near 1, the latent std positive."""
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name.endswith(FIXED):
                continue
            noise = torch.randn(tensor.shape, generator=torch.Generator().manual_seed(zlib.crc32(name.encode())))
            if name.endswith("norm.weight") and tensor.ndim == 1:
                tensor.copy_(1 + 0.1 * noise)
            elif name.endswith("latent_std"):
                tensor.copy_(1 + 0.1 * noise.abs())
            else:
                tensor.copy_(scale * noise)


def tiny_transformer(config: VisageConfig = TINY) -> VisageTransformer:
    model = VisageTransformer(config)
    fill(model)
    return model.eval()


def tiny_vae() -> VisageVAE:
    vae = VisageVAE(TINY_VAE)
    fill(vae)
    return vae.eval()


def tiny_inputs(h: int = 3, w: int = 2, length: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return (
        torch.randn(1, LATENT, h, w, generator=generator),
        torch.tensor([[0.35]]),
        torch.randn(1, length, CAP, generator=generator),
    )
