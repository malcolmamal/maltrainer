"""The Qwen-Image 2.1 LoRA trainer's step plan, its timestep draw, and one batch's flow-matching loss."""

from __future__ import annotations

import math
from typing import Any

import torch

from maltrainer.models.qwen_image_2_1.model import apply_flow_shift, calculate_shift
from maltrainer.utils.batching import batch_text_embeddings


def plan_training_steps(num_batches: int, train_config: dict[str, Any]) -> tuple[int, int, int]:
    """Return (steps_per_epoch, num_epochs, total_steps) for a run.

    ``train.max_steps`` (> 0) wins over ``train.num_epochs``: the run stops exactly at that
    many optimizer steps, and the epoch count is whatever it takes to get there (the last
    epoch may be partial). Otherwise the run is ``num_epochs`` full epochs.
    """
    grad_accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)
    steps_per_epoch = max(math.ceil(num_batches / grad_accum), 1)
    max_steps = int(train_config.get("max_steps", 0) or 0)
    if max_steps > 0:
        return steps_per_epoch, math.ceil(max_steps / steps_per_epoch), max_steps
    num_epochs = max(int(train_config.get("num_epochs", 10) or 1), 1)
    return steps_per_epoch, num_epochs, num_epochs * steps_per_epoch


def sample_timesteps(train_config: dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
    """Raw timesteps (before the size shift): sigmoid(N(0, sigmoid_scale)) clamped to [0.001, 0.999], or uniform."""
    distribution = str(train_config.get("timestep_distribution", "shift")).lower()
    if distribution in {"sigmoid", "shift", "logit_normal"}:
        scale = float(train_config.get("sigmoid_scale", 1.0))
        z = torch.randn(batch_size, device=device) * scale
        return torch.sigmoid(z).clamp(0.001, 0.999)
    return torch.rand(batch_size, device=device)


def batch_inputs(
    batch: dict[str, Any],
    caches: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The batch's pre-cached latents [B, 64, H_lat, W_lat] and text embeddings [B, text_len, 4096]."""
    latent_cache, text_cache = caches
    latents = torch.cat([latent_cache[p] for p in batch["image_path"]], dim=0).to(device=device, dtype=dtype)
    text_embeds = batch_text_embeddings([text_cache[c] for c in batch["caption"]])
    return latents, text_embeds.to(device=device, dtype=dtype)


def flow_loss(
    network: Any, latents: torch.Tensor, text_embeds: torch.Tensor, raw_timesteps: torch.Tensor
) -> torch.Tensor:
    """The MSE between the predicted and the target velocity, at the timesteps shifted for the image size."""
    bsz, c_lat, h_lat, w_lat = latents.shape
    img_tokens = h_lat * w_lat
    shifted_sigmas = apply_flow_shift(raw_timesteps, mu=calculate_shift(img_tokens))

    # Flow matching interpolation: x_t = (1 - t) * x_0 + t * noise, target velocity v = noise - x_0
    noise = torch.randn_like(latents)
    sigma_broadcast = shifted_sigmas.view(bsz, 1, 1, 1)
    noisy_latents = ((1.0 - sigma_broadcast) * latents + sigma_broadcast * noise).to(latents.dtype)
    target_velocity = noise - latents

    # Flatten latents: [B, 64, H, W] -> [B, H*W, 64]
    noisy_flat = noisy_latents.permute(0, 2, 3, 1).reshape(bsz, img_tokens, c_lat)
    target_flat = target_velocity.permute(0, 2, 3, 1).reshape(bsz, img_tokens, c_lat)
    pred_velocity = network(
        hidden_states=noisy_flat,
        encoder_hidden_states=text_embeds,
        timestep=shifted_sigmas,
        latent_h=h_lat,
        latent_w=w_lat,
    )
    return torch.nn.functional.mse_loss(pred_velocity.float(), target_flat.float())
