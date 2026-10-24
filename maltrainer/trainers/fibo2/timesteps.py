"""Fibo 2.0's training sigmas: BRIA's fine-tuning recipe by default, a uniform or logit-normal draw on request."""

from __future__ import annotations

from typing import Any

import torch

from maltrainer.training.sigma import LOGIT_NORMAL_NAMES, clamped, logit_normal, shifted

# BRIA's fine_tuning/train_lora.py: sigma = sigmoid(N(shift, 1)) with probability 0.9, else U(0, 1), where shift
# rises linearly with the number of target tokens n from 0.5 (at 256) to 1.15 (at 4096), the Flux2 scheduler's mu.
BASE_SHIFT, MAX_SHIFT = 0.5, 1.15
BASE_TOKENS, MAX_TOKENS = 256, 4096
UNIFORM_MIX = 0.1
DISTRIBUTIONS = ("bria", "uniform", "logit_normal")


def token_shift(tokens: int) -> float:
    """The mean of the logit-normal for a target of ``tokens`` latent pixels."""
    return BASE_SHIFT + (MAX_SHIFT - BASE_SHIFT) * (tokens - BASE_TOKENS) / (MAX_TOKENS - BASE_TOKENS)


def sample_sigma(train_config: dict[str, Any], tokens: int, device: torch.device) -> torch.Tensor:
    """Flow-matching sigma for one sample of ``tokens`` latent pixels (sigma 1 = noise)."""
    distribution = str(train_config.get("timestep_distribution", "bria")).lower()
    if distribution == "bria":
        mix = float(train_config.get("timestep_uniform_mix", UNIFORM_MIX) or 0.0)
        if mix > 0 and torch.rand((), device=device).item() < mix:
            base = torch.rand(1, device=device)
        else:
            base = torch.sigmoid(torch.randn(1, device=device) + token_shift(tokens))
    elif distribution in LOGIT_NORMAL_NAMES:
        base = logit_normal(train_config, device)
    elif distribution == "uniform":
        base = torch.rand(1, device=device)
    else:
        raise ValueError(f"train.timestep_distribution must be one of {DISTRIBUTIONS}, got {distribution!r}")
    return clamped(shifted(base, float(train_config.get("timestep_shift", 1.0))), train_config)
