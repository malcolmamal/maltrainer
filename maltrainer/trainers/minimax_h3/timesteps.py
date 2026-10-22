"""The MiniMax H3 LoRA trainer's video sigmas: the base draw, and the shift for the latent size."""

from __future__ import annotations

import math
from typing import Any

import torch


def sample_timesteps(train_config: dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
    # fp32: in bf16 a sigma near 1 lands on a 1/256 grid, and the audio remap
    # amplifies that step several-fold at the low-noise end of the audio schedule.
    distribution = str(train_config.get("timestep_distribution", "krea2_shift")).lower()
    if distribution in {"sigmoid", "krea2_shift", "krea2"}:
        # Logit-normal base, as musubi-tuner's krea2_shift does before shifting.
        scale = float(train_config.get("sigmoid_scale", 1.0))
        z = torch.randn(batch_size, device=device, dtype=torch.float32) * scale
        return torch.sigmoid(z).clamp(0.001, 0.999)
    if distribution in {"logit_normal", "logit-normal", "logitnormal"}:
        mean = float(train_config.get("timestep_logit_mean", 0.0))
        std = float(train_config.get("timestep_logit_std", 1.0))
        z = torch.randn(batch_size, device=device, dtype=torch.float32) * std + mean
        return torch.sigmoid(z).clamp(0.001, 0.999)
    return torch.rand(batch_size, device=device, dtype=torch.float32).clamp(0.001, 0.999)


def resolve_flow_shift(train_config: dict[str, Any], latent_h: int, latent_w: int) -> float:
    """Sigma shift for this latent size.

    H3's 12.0 is its *video* flow shift. Applied to still images it pushes ~57% of
    samples above sigma 0.9 and leaves only ~8% below 0.5, so almost all training
    happens in the high-noise range that sets composition rather than the low-noise
    range where facial identity and fine detail are resolved.

    krea2_shift instead derives the shift from the packed token count, matching
    musubi-tuner (x1=256 -> mu 0.5, x2=6400 -> mu 1.15, shift = exp(mu)). At 512px that
    is ~1.65, which puts ~31% of samples below sigma 0.5.
    """
    distribution = str(train_config.get("timestep_distribution", "krea2_shift")).lower()
    if distribution == "sigmoid":
        # No shift at all: sigma = sigmoid(N(0,1)), median 0.5. Community reports on
        # H3 character LoRAs single this out as important, paired with contrastive
        # guidance -- which changes the dynamics that made unshifted densities
        # overdrive adapters in trainers that ran them without it.
        return 1.0
    if distribution == "lownoise":
        # Fizgig's "Likeness and Style" default: uniform u through the shift map, with the
        # shift chosen so that exactly lownoise_pct % of steps land below sigma 0.5
        # (P(sigma < 0.5) = 1 / (1 + shift)). 60 % -> shift 0.667.
        share = float(train_config.get("lownoise_pct", 60)) / 100.0
        if not 0.0 < share < 1.0:
            raise ValueError("train.lownoise_pct must be between 0 and 100 (exclusive)")
        return (1.0 - share) / share
    if distribution not in {"krea2_shift", "krea2"}:
        return float(train_config.get("timestep_shift", 12.0))

    packed_tokens = (latent_h // 2) * (latent_w // 2)
    slope = (1.15 - 0.5) / (6400 - 256)
    mu = 0.5 + slope * (packed_tokens - 256)
    return math.exp(mu)
