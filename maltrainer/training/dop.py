"""Differential output preservation (ai-toolkit's diff_output_preservation).

On the same noised latent and sigma as the training pass, the sample's caption with the trigger replaced by
the class word: the base model's prediction for it (adapters off, no grad) is the target for the LoRA's.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def check_dop_config(config: dict[str, Any]) -> None:
    """train.dop needs the class word and the trigger it replaces."""
    train_config = config.get("train", {})
    if not train_config.get("dop"):
        return
    if not str(train_config.get("dop_class", "") or "").strip():
        raise ValueError("train.dop needs train.dop_class, the class word that replaces the trigger")
    if not str(config.get("dataset", {}).get("trigger", "") or "").strip():
        raise ValueError("train.dop needs dataset.trigger: DOP's captions are the captions without it")


def dop_captions(samples: list[tuple[str, str]], trigger: str, cls: str) -> dict[str, str]:
    """Sample key -> its caption with ``trigger`` replaced by ``cls``, for the captions that name the trigger."""
    captions = {key: caption.replace(trigger, cls) for key, caption in samples if trigger in caption}
    example = next(((c, captions[k]) for k, c in samples if k in captions), None)
    logger.info(
        "DOP: %d of %d captions name the trigger %r; their DOP caption swaps it for %r%s",
        len(captions),
        len(samples),
        trigger,
        cls,
        f" (e.g. {example[0]!r} -> {example[1]!r})" if example else "",
    )
    missing = len(samples) - len(captions)
    if missing:
        logger.warning(
            "DOP: %d caption(s) don't contain the trigger %r and get no DOP pass (holding them to the "
            "base model would unlearn the subject).",
            missing,
            trigger,
        )
    if not captions:
        raise ValueError(f"train.dop: no caption contains the trigger {trigger!r}")
    return captions


def dop_loss(network: Any, adapters: Any, inputs: tuple[torch.Tensor, ...], multiplier: float) -> torch.Tensor:
    """The MSE between the LoRA's prediction for ``inputs`` and the base model's (``adapters.disabled()``)."""
    with torch.no_grad(), adapters.disabled():
        target = network(*inputs).float()
    pred = network(*inputs)
    return torch.nn.functional.mse_loss(pred.float(), target) * multiplier
