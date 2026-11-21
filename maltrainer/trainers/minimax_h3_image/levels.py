"""Which of the checkpoint's distilled noise levels a training step runs at.

H3 Image renders in 6 fixed steps, each through its own output head, so a LoRA is trained at exactly those levels.
By default the levels are drawn evenly. "Likeness optimization" instead puts a share of the steps on the levels below
sigma 0.5 (the nearly finished picture, where skin, eyes and fine detail are made): 75 % for the first 8 epochs,
then 50 % and 75 % in turn, so the early steps (pose, layout) keep up.
"""

from __future__ import annotations

import torch

CLEAN_BELOW = 0.5  # levels under this sigma are the "clean end"
FIRST_EPOCHS = 8  # epochs at the full share before it starts alternating


def clean_share(epoch: int, likeness: bool) -> float | None:
    """The share of steps at the clean end for this (1-based) epoch; None for the even draw."""
    if not likeness:
        return None
    return 0.75 if epoch <= FIRST_EPOCHS or (epoch - FIRST_EPOCHS) % 2 == 0 else 0.5


def draw_level(levels: tuple[float, ...], share: float | None) -> int:
    """The index of the level (a sigma in ``levels``, which excludes the final 0) a step runs at."""
    clean = [i for i, sigma in enumerate(levels) if sigma < CLEAN_BELOW]
    noisy = [i for i, sigma in enumerate(levels) if sigma >= CLEAN_BELOW]
    if share is None or not clean or not noisy:
        return int(torch.randint(len(levels), (1,)))
    weights = torch.zeros(len(levels))
    weights[clean] = share / len(clean)
    weights[noisy] = (1.0 - share) / len(noisy)
    return int(torch.multinomial(weights, 1))
