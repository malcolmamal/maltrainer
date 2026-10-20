"""A tiny Krea 2 DiT, the real architecture at toy sizes (head dim 16 -> RoPE axes 4 + 6 + 6); a copy of the model
tests' own, so these tests do not depend on another test folder."""

from __future__ import annotations

import torch

from maltrainer.models.krea2.model import Krea2Config, Krea2Transformer

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
    for param in model.parameters():
        if not param.any():
            torch.nn.init.normal_(param, std=0.1)
    model.compute_dtype = torch.float32
    return model
