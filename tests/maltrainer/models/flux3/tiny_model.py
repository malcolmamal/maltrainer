"""A FLUX.3 DiT small enough for CPU tests: head_dim must stay 128 (four RoPE axes of 32), so one head of 128."""

from __future__ import annotations

import torch

from maltrainer.models.flux3.model import CONTEXT_DIM, LATENT_CHANNELS, Flux3Config, Flux3Transformer

TINY = Flux3Config(hidden_size=128, num_heads=1, depth=1, depth_single_blocks=2)


def tiny_transformer(config: Flux3Config = TINY, seed: int = 0) -> Flux3Transformer:
    torch.manual_seed(seed)
    model = Flux3Transformer(config)
    model.compute_dtype = torch.float32
    return model.eval()


def tiny_inputs(batch: int = 2, h: int = 2, w: int = 3, length: int = 4) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(batch, LATENT_CHANNELS, h, w, generator=generator)
    sigma = torch.linspace(0.25, 0.75, batch)
    context = torch.randn(batch, length, CONTEXT_DIM, generator=generator)
    return x, sigma, context
