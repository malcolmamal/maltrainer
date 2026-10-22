"""MOSS-Image's training sigmas; the defaults are the technical report's final pretraining interval."""

from __future__ import annotations

from typing import Any

import torch

from maltrainer.training.sigma import LOGIT_NORMAL_NAMES, clamped, shifted

# The technical report's last pretraining interval (Table 6, steps 274,001-310,000): sigma = 1 -
# sigmoid(u), u ~ N(-1.09, 0.95), i.e. logit(sigma) ~ N(1.09, 0.95); truncated at sigma <= 0.95,
# rescaled onto the 1000-step grid {0.001 .. 1}, mixed with 1 % discrete uniform; shift 1.
MOSS_LOGIT_MEAN = 1.09
MOSS_LOGIT_STD = 0.95
MOSS_CUTOFF = 0.95
MOSS_UNIFORM_MIX = 0.01
MOSS_GRID = 1000


def sample_sigma(train_config: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Flow-matching sigma for one sample.

    1. logit_normal: logit(sigma) ~ N(mean, std), truncated to sigma <= cutoff by inverse-CDF
       sampling and divided by the cutoff; or uniform
    2. with probability ``timestep_uniform_mix``, a uniform draw instead
    3. the shift sigma' = s * sigma / (1 + (s - 1) * sigma); 1 = none
    4. rounded up onto the training grid {1/N .. 1}, N = ``timestep_grid`` (0 = continuous)
    """
    distribution = str(train_config.get("timestep_distribution", "logit_normal")).lower()
    if distribution in LOGIT_NORMAL_NAMES:
        base = _logit_normal(train_config, device)
    elif distribution == "uniform":
        base = torch.rand(1, device=device)
    else:
        raise ValueError(f"train.timestep_distribution must be logit_normal or uniform, got {distribution!r}")
    mix = float(train_config.get("timestep_uniform_mix", MOSS_UNIFORM_MIX) or 0.0)
    if mix > 0 and torch.rand(()).item() < mix:
        base = torch.rand(1, device=device)
    sigma = shifted(base, float(train_config.get("timestep_shift", 1.0)))
    grid = int(train_config.get("timestep_grid", MOSS_GRID) or 0)
    if grid > 0:
        sigma = torch.ceil(sigma * grid).clamp(1, grid) / grid
    return clamped(sigma, train_config, max_sigma=1.0)


def _logit_normal(train_config: dict[str, Any], device: torch.device) -> torch.Tensor:
    mean = float(train_config.get("timestep_logit_mean", MOSS_LOGIT_MEAN))
    std = float(train_config.get("timestep_logit_std", MOSS_LOGIT_STD))
    cutoff = float(train_config.get("timestep_cutoff", MOSS_CUTOFF))
    if cutoff < 1.0:
        top = torch.special.ndtr(torch.tensor((torch.logit(torch.tensor(cutoff)).item() - mean) / std))
        p = torch.rand(1, device=device) * top.to(device)
        return torch.sigmoid(mean + std * torch.special.ndtri(p)) / cutoff
    return torch.sigmoid(torch.randn(1, device=device) * std + mean)
