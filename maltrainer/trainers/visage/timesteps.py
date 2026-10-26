"""Visage-Image's training sigmas: the release recipe by default, a plain logit-normal or a uniform draw on request."""

from __future__ import annotations

import math
from typing import Any

import torch

from maltrainer.training.sigma import LOGIT_NORMAL_NAMES, clamped, logit_normal, shifted

# configs/dit.yaml: t = sigmoid(N(0, 1) + log(sqrt(tokens / 64))), so the mean of the logit is 0.5 * ln(tokens / 64),
# 0.69 at the 256 tokens of a 256 px image: more tokens are more redundant, so the draw moves toward noise.
SHIFT_REF_TOKENS = 64
DISTRIBUTIONS = ("visage", "uniform", "logit_normal")


def logit_mean(tokens: int, ref_tokens: int = SHIFT_REF_TOKENS) -> float:
    """The mean of the logit-normal for an image of ``tokens`` latent cells."""
    return 0.5 * math.log(tokens / ref_tokens)


def sample_sigma(train_config: dict[str, Any], tokens: int, device: torch.device) -> torch.Tensor:
    """Flow-matching sigma for one sample of ``tokens`` latent cells (sigma 1 = noise)."""
    distribution = str(train_config.get("timestep_distribution", "visage")).lower()
    if distribution == "visage":
        ref = int(train_config.get("timestep_ref_tokens", SHIFT_REF_TOKENS))
        base = logit_normal(train_config, device, mean=logit_mean(tokens, ref))
    elif distribution in LOGIT_NORMAL_NAMES:
        base = logit_normal(train_config, device)
    elif distribution == "uniform":
        base = torch.rand(1, device=device)
    else:
        raise ValueError(f"train.timestep_distribution must be one of {DISTRIBUTIONS}, got {distribution!r}")
    return clamped(shifted(base, float(train_config.get("timestep_shift", 1.0))), train_config)
