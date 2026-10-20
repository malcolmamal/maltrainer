"""Krea 2's training sigmas: a logit-normal (or uniform) draw, shifted by the image's token count by default."""

from __future__ import annotations

import math
from typing import Any

import torch

from maltrainer.training.sigma import LOGIT_NORMAL_NAMES, clamped, logit_normal, shifted


def krea2_shift(image_tokens: int) -> float:
    """Resolution-dependent sigma shift (musubi-tuner / Fizgig): mu 0.5 at 256 tokens, 1.15 at 6400."""
    mu = 0.5 + (image_tokens - 256) * (1.15 - 0.5) / (6400 - 256)
    return math.exp(mu)


def sample_sigma(train_config: dict[str, Any], image_tokens: int, device: torch.device) -> tuple[torch.Tensor, float]:
    """Video-free flow-matching sigma for one sample, plus the shift used: krea2_shift's for the default
    ``krea2_shift`` distribution, none for ``sigmoid``, else ``train.timestep_shift``."""
    distribution = str(train_config.get("timestep_distribution", "krea2_shift")).lower()
    if distribution in LOGIT_NORMAL_NAMES | {"krea2_shift"}:
        base = logit_normal(train_config, device)
    elif distribution == "uniform":
        base = torch.rand(1, device=device)
    else:
        raise ValueError(
            f"train.timestep_distribution must be krea2_shift, logit_normal, sigmoid or uniform: {distribution!r}"
        )
    if distribution == "krea2_shift":
        shift = krea2_shift(image_tokens)
    elif distribution == "sigmoid":
        shift = 1.0
    else:
        shift = float(train_config.get("timestep_shift", 1.0))
    return clamped(shifted(base, shift), train_config), shift
