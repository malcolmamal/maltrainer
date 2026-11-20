"""Masked training's loss: a mask weights each latent position's squared error, and the part outside it can be held to
the base model's own prediction (masked prior preservation: OneTrainer, AI-Toolkit's inverted_mask_prior).

A picture counts the same however little of it is white: the weights are scaled to mean 1 per picture.
"""

from __future__ import annotations

import torch


def mask_weight(mask: torch.Tensor, floor: float) -> torch.Tensor:
    """A mask in [0, 1] as loss weights: never below the floor, 1 where it is white."""
    return mask.clamp(min=float(floor), max=1.0)


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    """The mean squared error under ``weight`` ([1, 1, h, w], scaled to mean 1); plain MSE without one."""
    error = (pred.float() - target.float()).pow(2)
    if weight is None:
        return error.mean()
    weight = weight.to(error.device, error.dtype)
    return (error * (weight / weight.mean().clamp_min(1e-8))).mean()


def prior_loss(pred: torch.Tensor, base: torch.Tensor, weight: torch.Tensor, prior: float) -> torch.Tensor:
    """Outside the mask (1 - its weight: black counts 1 - floor) the prediction is pulled to ``base``, the model's own
    with the LoRA at zero, at ``prior`` x the inside's weight; normalised over that area as the inside is over its
    own."""
    outside = (1.0 - weight).clamp_min(0.0)
    return float(prior) * weighted_mse(pred, base, outside)
