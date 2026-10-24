"""One LTX-2.5 training step: noise the video (and the soundtrack, if the clip has one) and score the velocities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from maltrainer.trainers.ltx25.timesteps import sample_sigma


def flow_inputs(clean: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(noisy latent, velocity target) of rectified flow: x_t = (1 - sigma) x0 + sigma noise, v = noise - x0."""
    noise = torch.randn_like(clean)
    return (1.0 - sigma) * clean + sigma * noise, noise - clean


def ltx_loss(
    network: Any,
    train_config: dict[str, Any],
    video: torch.Tensor,
    features: torch.Tensor,
    audio: torch.Tensor | None,
    fps: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """video [1, 128, F, H, W] and audio [1, 8, T, 16] (or None), both clean latents in fp32; features [1, N, D]."""
    device = video.device
    sigma = sample_sigma(train_config, video.shape[2] * video.shape[3] * video.shape[4], device)
    noisy_video, video_target = flow_inputs(video, sigma.view(1, 1, 1, 1, 1))
    noisy_audio = audio_target = None
    if audio is not None:
        noisy_audio, audio_target = flow_inputs(audio, sigma.view(1, 1, 1, 1))
    video_pred, audio_pred = network(
        noisy_video.to(dtype), sigma, features, fps, None if noisy_audio is None else noisy_audio.to(dtype), sigma
    )
    loss = float(train_config.get("video_loss_weight", 1.0)) * F.mse_loss(video_pred.float(), video_target)
    if audio_pred is not None and audio_target is not None:
        loss = loss + float(train_config.get("audio_loss_weight", 1.0)) * F.mse_loss(audio_pred.float(), audio_target)
    return loss
