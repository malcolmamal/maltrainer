"""The Z-Image trainers' optional latent and noise transforms (AI-Toolkit style), and the timestep draw."""

from __future__ import annotations

from typing import Any

import torch


def transformed(
    latents: torch.Tensor, noise: torch.Tensor, train_config: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """(latents, noise) after ``latent_multiplier``, ``noise_multiplier``, ``dynamic_noise_offset``,
    ``random_noise_multiplier`` and ``random_noise_shift``, in that order."""
    latent_multiplier = float(train_config.get("latent_multiplier", 1.0))
    if latent_multiplier != 1.0:
        latents = latents * latent_multiplier

    noise_multiplier = float(train_config.get("noise_multiplier", 1.0))
    if noise_multiplier != 1.0:
        noise = noise * noise_multiplier

    if train_config.get("dynamic_noise_offset", False):
        latents_channel_mean = latents.mean(dim=(2, 3), keepdim=True) / 2
        noise = noise + latents_channel_mean

    random_noise_multiplier = float(train_config.get("random_noise_multiplier", 0.0))
    if random_noise_multiplier > 0.0:
        s = (noise.shape[0], noise.shape[1], 1, 1)
        noise = noise * (1 + torch.randn(s, device=noise.device, dtype=noise.dtype) * random_noise_multiplier)

    random_noise_shift = float(train_config.get("random_noise_shift", 0.0))
    if random_noise_shift > 0.0:
        s = (noise.shape[0], noise.shape[1], 1, 1)
        noise = noise + torch.randn(s, device=noise.device, dtype=noise.dtype) * random_noise_shift
    return latents, noise


def sample_timesteps(
    train_config: dict[str, Any], batch_size: int, num_train_timesteps: int, on: tuple[torch.device, torch.dtype]
) -> torch.Tensor:
    """Uniform or logit-normal on [0, 1] (0 clean, 1 noise), times ``timestep_shift`` (clamped), snapped onto the
    scheduler's grid."""
    device, dtype = on
    distribution = str(train_config.get("timestep_distribution", "uniform")).lower()
    if distribution in {"logit_normal", "logit-normal", "logitnormal"}:
        mean = float(train_config.get("timestep_logit_mean", 0.0))
        std = float(train_config.get("timestep_logit_std", 1.0))
        z = torch.randn(batch_size, device=device, dtype=dtype) * std + mean
        timesteps = torch.sigmoid(z).clamp(0.0, 1.0)
    else:
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)

    shift = float(train_config.get("timestep_shift", 1.0))
    if shift != 1.0:
        timesteps = (timesteps * shift).clamp(0.0, 1.0)

    if num_train_timesteps > 0:
        timesteps = (timesteps * num_train_timesteps).floor() / num_train_timesteps
    return timesteps
