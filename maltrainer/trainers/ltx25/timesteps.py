"""LTX-2.x's training sigmas: a logit-normal whose mean grows with the number of video tokens (the official trainer)."""

from __future__ import annotations

from typing import Any

import torch

from maltrainer.training.sigma import clamped

# Official LTX-2 trainer: the logit-normal's mean (the "shift") is linear in the video token count, 0.95 at 1024
# tokens and 2.05 at 4096, extrapolated beyond. "stretched" (LTX-2.3 and 2.5) then maps the 0.5 .. 99.9 percentiles of
# the distribution onto [0, 1], reflects values below eps and replaces a tenth of the draws with uniform ones.
MIN_TOKENS, MAX_TOKENS = 1024, 4096
MIN_SHIFT, MAX_SHIFT = 0.95, 2.05
P999_Z, P005_Z = 3.0902, -2.5758
DISTRIBUTIONS = ("shifted_logit_normal", "uniform")
MODES = ("stretched", "legacy")


def token_shift(tokens: int) -> float:
    slope = (MAX_SHIFT - MIN_SHIFT) / (MAX_TOKENS - MIN_TOKENS)
    return slope * (tokens - MIN_TOKENS) + MIN_SHIFT


def shifted_logit_normal(
    shift: float, std: float, mode: str, eps: float, uniform_prob: float, device: torch.device
) -> torch.Tensor:
    base = torch.sigmoid(torch.randn(1, device=device) * std + shift)
    if mode == "legacy":
        return base
    low = torch.sigmoid(torch.tensor(shift + P005_Z * std, device=device))
    high = torch.sigmoid(torch.tensor(shift + P999_Z * std, device=device))
    stretched = (base - low) / (high - low).clamp(min=1e-6)
    stretched = torch.where(stretched >= eps, stretched, 2 * eps - stretched).clamp(0.0, 1.0)
    if uniform_prob > 0 and torch.rand((), device=device).item() < uniform_prob:
        return (1.0 - eps) * torch.rand(1, device=device) + eps
    return stretched


def sample_sigma(train_config: dict[str, Any], tokens: int, device: torch.device) -> torch.Tensor:
    """Flow-matching sigma for one sample of ``tokens`` video latent tokens (sigma 1 = noise)."""
    distribution = str(train_config.get("timestep_distribution", "shifted_logit_normal")).lower()
    if distribution == "uniform":
        return clamped(torch.rand(1, device=device), train_config, max_sigma=1.0)
    if distribution != "shifted_logit_normal":
        raise ValueError(f"train.timestep_distribution must be one of {DISTRIBUTIONS}, got {distribution!r}")
    mode = str(train_config.get("shifted_logit_mode", "stretched")).lower()
    if mode not in MODES:
        raise ValueError(f"train.shifted_logit_mode must be one of {MODES}, got {mode!r}")
    shift = float(train_config.get("shifted_logit_shift", token_shift(tokens)))
    sigma = shifted_logit_normal(
        shift,
        float(train_config.get("logit_std", 1.0)),
        mode,
        min(max(float(train_config.get("shifted_logit_eps", 1e-3)), 0.0), 0.499),
        min(max(float(train_config.get("shifted_logit_uniform_prob", 0.1)), 0.0), 1.0),
        device,
    )
    return clamped(sigma, train_config, max_sigma=1.0)
