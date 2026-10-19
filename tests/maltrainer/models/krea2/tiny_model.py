"""A Krea 2 DiT small enough for CPU tests, plus synthetic inputs for it."""

from __future__ import annotations

import torch

from maltrainer.models.krea2.model import Krea2Config, Krea2Transformer

TINY = Krea2Config(
    features=32,
    tdim=16,
    txtdim=32,
    heads=2,
    kvheads=1,
    multiplier=1,
    layers=2,
    patch=2,
    channels=4,
    txtlayers=3,
    txtheads=2,
    txtkvheads=1,
)


def tiny_transformer(config: Krea2Config = TINY, seed: int = 0) -> Krea2Transformer:
    torch.manual_seed(seed)
    model = Krea2Transformer(config)
    for parameter in model.parameters():
        parameter.data.normal_(std=0.2)
    model.compute_dtype = torch.float32
    return model.eval()


def tiny_inputs(batch: int = 2, h: int = 4, w: int = 6, length: int = 3) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(batch, TINY.channels, h, w, generator=generator)
    sigma = torch.linspace(0.25, 0.75, batch)
    context = torch.randn(batch, length, TINY.txtlayers, TINY.txtdim, generator=generator)
    return x, sigma, context
