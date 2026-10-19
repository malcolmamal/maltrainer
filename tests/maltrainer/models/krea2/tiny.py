"""A tiny Krea 2 DiT: the real architecture at toy sizes (head dim 16 -> RoPE axes 4 + 6 + 6)."""

from __future__ import annotations

import torch

from maltrainer.models.krea2.model import Krea2Config, Krea2Transformer

CPU = torch.device("cpu")
CONFIG = Krea2Config(
    features=64,
    tdim=16,
    txtdim=32,
    heads=4,
    kvheads=2,
    multiplier=2,
    layers=2,
    patch=2,
    channels=4,
    theta=1e3,
    txtlayers=3,
    txtheads=2,
    txtkvheads=2,
)


def tiny_model(seed: int = 0) -> Krea2Transformer:
    """Float32 weights with every zero-initialised parameter randomised, so no path is dead."""
    torch.manual_seed(seed)
    model = Krea2Transformer(CONFIG)
    for name, param in model.named_parameters():
        if not param.any():
            torch.nn.init.normal_(param, std=0.1)
    model.compute_dtype = torch.float32
    return model


def inputs(batch: int = 2, size: int = 8, seq: int = 3, seed: int = 1):
    """(latent, sigma, context) for the tiny model: 4 channels, ``size`` x ``size``, 3 text layers."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, CONFIG.channels, size, size, generator=generator)
    sigma = torch.rand(batch, generator=generator)
    context = torch.randn(batch, seq, CONFIG.txtlayers, CONFIG.txtdim, generator=generator)
    return x, sigma, context
