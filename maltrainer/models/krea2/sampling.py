"""
Krea 2 sampling for training previews: flow-matching Euler with CFG, as Fizgig samples the RAW model
(fizgig/krea2/sampling.py, after ai-toolkit's flux2 sampler; Fizgig's RAW settings: 28 steps, CFG 4.5).

- start: fp32 noise [1, 16, H/8, W/8] from a CPU generator seeded with the seed, so a seed renders the same picture
  on any device and every time
- schedule: ``steps`` + 1 points from 1 to 0, shifted by the image's token count with the training shift
  (``timesteps.krea2_shift``: mu 0.5 at 256 tokens to 1.15 at 6400), ``s t / (1 + (s - 1) t)``
- Euler on the velocity v = noise - x0, in fp32: x <- x + (sigma_next - sigma) v
- CFG ``cfg`` against the negative prompt's embedding: v = v_neg + cfg (v - v_neg)

The DiT takes sigma in [0, 1] and [1, seq, 12, 2560] text; any height and width that are multiples of 16 work
(VAE 8x, patch 2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from maltrainer.models.krea2.timesteps import krea2_shift

CHANNELS = 16  # Krea2Config.channels
DOWNSAMPLING = 8  # the VAE
ALIGN = 16  # VAE 8x, patch 2


def schedule(steps: int, image_tokens: int) -> torch.Tensor:
    """[steps + 1] fp32 sigmas from 1 to 0, shifted for an image of ``image_tokens`` tokens."""
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    shift = krea2_shift(image_tokens)
    t = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return (shift * t / (1.0 + (shift - 1.0) * t)).float()


def size_problem(width: int, height: int) -> str | None:
    """What is wrong with a preview size, with the nearest sizes that work; None when it is fine."""
    bad = [n for n in (width, height) if n <= 0 or n % ALIGN]
    if not bad:
        return None
    hints = ", ".join(
        f"{n // ALIGN * ALIGN} or {(n // ALIGN + 1) * ALIGN} for {n}" if n > ALIGN else f"at least {ALIGN} for {n}"
        for n in dict.fromkeys(bad)
    )
    return f"sample.width / height must be multiples of {ALIGN}, got {width}x{height}: use {hints}"


def initial_noise(seed: int, height: int, width: int, channels: int = CHANNELS) -> torch.Tensor:
    problem = size_problem(width, height)
    if problem:
        raise ValueError(problem)
    generator = torch.Generator("cpu").manual_seed(int(seed))
    shape = (1, channels, height // DOWNSAMPLING, width // DOWNSAMPLING)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _batched(text: torch.Tensor, device: torch.device) -> torch.Tensor:
    return (text if text.dim() == 4 else text[None]).to(device)


@torch.no_grad()
def sample(
    dit: Callable[..., Any],
    context: torch.Tensor,
    height: int,
    width: int,
    *,
    steps: int = 28,
    seed: int = 42,
    cfg: float = 4.5,
    negative: torch.Tensor | None = None,
    channels: int = CHANNELS,
    device: torch.device | None = None,
    on_step: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Denoised normalized latents [1, channels, H/8, W/8] fp32. ``dit(x, sigma [1], context)`` is the DiT; context /
    negative: [L, 12, 2560] or [1, L, 12, 2560]. CFG needs ``negative`` and ``cfg`` != 1. ``on_step(done, total)``
    is called after every step (a progress bar)."""
    device = device or context.device
    x = initial_noise(seed, height, width, channels).to(device)
    pos = _batched(context, device)
    neg = _batched(negative, device) if negative is not None and cfg != 1.0 else None
    sigmas = schedule(steps, (x.shape[-2] // 2) * (x.shape[-1] // 2))
    for i in range(steps):
        s, s_next = float(sigmas[i]), float(sigmas[i + 1])
        sigma = torch.tensor([s], device=device)
        v = dit(x, sigma, pos).float()
        if neg is not None:
            vu = dit(x, sigma, neg).float()
            v = vu + cfg * (v - vu)
        x = x + (s_next - s) * v
        if on_step is not None:
            on_step(i + 1, steps)
    return x
