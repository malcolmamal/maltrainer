"""
Regularisation images for a full finetune: ordinary photos of the broader class, trained beside the
subjects at a reduced step size as an anchor (Fizgig's "Regularisation images", LR x 0.2).

A finetune has no low-rank bound, so a long run on a few people drifts the model's whole idea of a
person. Class photos pull it back. At ``lr_multiplier`` 0.1-0.3 they tether the prior; toward 1.0
they train as a second subject set instead.

The multiplier scales the optimizer's step, not the loss: Automagic3 (like Adafactor) divides each
update by its own RMS, so a scaled loss would come out at almost the same step size. Use real
photos, not the model's own output, or the anchor distils its artifacts back in.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence

import torch

logger = logging.getLogger(__name__)


def epoch_order(subjects: Sequence[int], reg: Sequence[int], repeats: int, reg_repeats: int) -> list[int]:
    """One epoch's shuffled sample indices: every subject ``repeats`` times, every regularisation
    image ``reg_repeats`` times. Uses Python's ``random`` (a resumed run restores it)."""
    order = [i for i in subjects for _ in range(max(int(repeats), 1))]
    order += [i for i in reg for _ in range(max(int(reg_repeats), 0))]
    random.shuffle(order)
    return order


def set_step_scale(optimizer: torch.optim.Optimizer, scale: float) -> None:
    """Scale the size of the coming update(s). Automagic3 reads ``lr_scale`` in every per-parameter
    update, fused (inside backward) or not; set it before backward when fused, before ``step()``
    otherwise."""
    inner = getattr(optimizer, "optimizer", optimizer)
    if not hasattr(inner, "lr_scale"):
        raise ValueError(f"{type(inner).__name__} has no lr_scale; regularisation images need Automagic3")
    inner.lr_scale = float(scale)
