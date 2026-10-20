"""Flow-matching sigma draws the trainers share: the logit-normal base, the shift and the clamp."""

from __future__ import annotations

from typing import Any

import torch

LOGIT_NORMAL_NAMES = {"logit_normal", "logit-normal", "sigmoid"}


def logit_normal(
    train_config: dict[str, Any], device: torch.device, mean: float = 0.0, std: float = 1.0
) -> torch.Tensor:
    """One draw with logit(sigma) ~ N(``timestep_logit_mean``, ``timestep_logit_std``)."""
    mean = float(train_config.get("timestep_logit_mean", mean))
    std = float(train_config.get("timestep_logit_std", std))
    return torch.sigmoid(torch.randn(1, device=device) * std + mean)


def shifted(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """The flow shift s * sigma / (1 + (s - 1) * sigma); 1 leaves sigma as it is."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def clamped(sigma: torch.Tensor, train_config: dict[str, Any], max_sigma: float = 0.999) -> torch.Tensor:
    """Into [``min_sigma``, ``max_sigma``]."""
    return sigma.clamp(float(train_config.get("min_sigma", 0.001)), float(train_config.get("max_sigma", max_sigma)))
