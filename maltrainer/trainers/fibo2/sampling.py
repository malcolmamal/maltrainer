"""Fibo 2.0 sampling for training previews: BRIA's pipeline, a shifted flow-matching Euler with CFG.

Mirrors diffusers' ``BriaFibo2Pipeline`` (and ComfyUI's Fibo 2.0 sampler): sigmas run from 1 to 1 / steps under the
FLUX.2 resolution-dependent exponential shift (mu 0.5 at 256 tokens to 1.15 at 4096), then 0; fp32 Euler steps on
fp32 noise; the guided velocity is ``uncond + guidance * (cond - uncond)`` against the null caption's features.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from maltrainer.trainers.fibo2.timesteps import BASE_SHIFT, BASE_TOKENS, MAX_SHIFT, MAX_TOKENS

DEFAULT_STEPS, DEFAULT_GUIDANCE = 30, 5.0


def sampling_sigmas(steps: int, tokens: int) -> torch.Tensor:
    """The ``steps + 1`` sigmas (the last is 0) for an image of ``tokens`` latent pixels."""
    slope = (MAX_SHIFT - BASE_SHIFT) / (MAX_TOKENS - BASE_TOKENS)
    mu = tokens * slope + BASE_SHIFT - slope * BASE_TOKENS
    base = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / base - 1.0))
    return torch.cat((shifted.float(), torch.zeros(1)))


@torch.no_grad()
def sample_latents(
    transformer: Any,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    grid: tuple[int, int],
    steps: int,
    guidance: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Denoised latents [1, 128, h, w] for ``grid`` = (h, w); ``cond`` and ``uncond`` are [1, T, D] text features."""
    height, width = grid
    noise = torch.randn((1, 128, height, width), generator=torch.Generator().manual_seed(seed))
    latents = noise.to(device)
    sigmas = sampling_sigmas(steps, height * width)
    for index in range(steps):
        sigma = sigmas[index].to(device).reshape(1)
        velocity = transformer(latents.to(dtype), sigma, cond).float()
        if guidance != 1.0:
            plain = transformer(latents.to(dtype), sigma, uncond).float()
            velocity = plain + guidance * (velocity - plain)
        latents = latents + (sigmas[index + 1] - sigmas[index]).item() * velocity
    return latents
