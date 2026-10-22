"""The Z-Image LoRA trainer's loss: per-sample MSE against latents - noise (Musubi's target), weighted by timestep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

# AI-Toolkit's timestep weighting curve ("aitk"), indexed by the timestep's place on [0, 1].
AITK_WEIGHTING: list[float] = json.loads((Path(__file__).parent / "aitk_weighting.json").read_text(encoding="utf-8"))


def timestep_weights(timesteps: torch.Tensor, weighting: str) -> torch.Tensor | None:
    """Per-sample weights for ``train.timestep_weighting`` (None: unweighted)."""
    if weighting == "linear":
        return 0.5 + timesteps
    if weighting == "sqrt":
        return 0.5 + torch.sqrt(timesteps)
    if weighting == "cosine":
        return 0.5 + torch.cos((1.0 - timesteps) * torch.pi * 0.5)
    if weighting == "aitk":
        w = torch.tensor(AITK_WEIGHTING, device=timesteps.device, dtype=timesteps.dtype)
        idx = (timesteps * (w.numel() - 1)).long().clamp(0, w.numel() - 1)
        return w[idx]
    return None


def weighted_loss(
    model_pred: torch.Tensor, target: torch.Tensor, timesteps: torch.Tensor, train_config: dict[str, Any]
) -> torch.Tensor:
    loss_map = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
    per_sample_loss = loss_map.mean(dim=(1, 2, 3))
    weights = timestep_weights(timesteps, str(train_config.get("timestep_weighting", "none")).lower())
    loss = (per_sample_loss * weights).mean() if weights is not None else per_sample_loss.mean()
    if str(train_config.get("loss_weight_fn", "none")).lower() == "constant":
        loss = loss * float(train_config.get("loss_weight_strength", 1.0))
    return loss
