"""The Qwen-Image 2.1 LoRA trainer's optimizer: its own defaults (AdamW8bit, weight decay 0.01), a fall back
to AdamW when bitsandbytes or Prodigy cannot be used, and its warmup from a zero rate."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.optim import AdamW

from maltrainer.optimizers.prodigy import build_prodigy
from maltrainer.training.optimizer import automagic_one_group, eight_bit_or_adamw, linear_warmup

logger = logging.getLogger(__name__)

Resolved = dict[str, object] | None  # what the metadata records; None after a fallback


def warmup_steps(train_config: dict[str, Any]) -> int:
    """``train.warmup_steps``, or the other trainers' ``learning_rate_warmup_steps``."""
    return int(train_config.get("warmup_steps") or train_config.get("learning_rate_warmup_steps") or 0)


def build_optimizer(params: list[torch.nn.Parameter], train_config: dict[str, Any]) -> tuple[Any, Resolved]:
    """(optimizer, what it resolved to) for ``train.optimizer``."""
    lr = float(train_config.get("learning_rate", 1e-4))
    name = str(train_config.get("optimizer", "adamw8bit")).lower()
    weight_decay = float(train_config.get("weight_decay", 0.01))
    if name in {"adamw8bit", "adam8bit"}:
        return eight_bit_or_adamw(params, name, lr, weight_decay)
    if name.startswith("prodigy"):
        return _prodigy(params, train_config, lr, weight_decay)
    if name in {"automagic3", "automagic", "automagic_v3"}:
        return automagic_one_group(params, train_config, weight_decay)
    return AdamW(params, lr=lr, weight_decay=weight_decay), {"name": "adamw", "lr": lr}


def _prodigy(
    params: list[torch.nn.Parameter], train_config: dict[str, Any], lr: float, weight_decay: float
) -> tuple[Any, Resolved]:
    try:
        recipe = {  # this trainer's defaults (the shared ones suit the other LoRA trainers)
            "prodigy_beta2": 0.99,
            "prodigy_use_bias_correction": True,
            "prodigy_safeguard_warmup": warmup_steps(train_config) > 0,
            "prodigy_slice_p": None,
        }
        configured_lr = float(train_config.get("learning_rate", 1.0))
        optimizer, supported = build_prodigy(params, train_config, configured_lr, weight_decay, recipe)
        logger.info(
            "Using Prodigy optimizer: mult=%.2f, d_coef=%.2f, d0=%.1e",
            supported["lr"],
            supported.get("d_coef", 1.0),
            supported.get("d0", 1e-6),
        )
        return optimizer, {"name": "prodigy", **supported}
    except (ImportError, OSError, RuntimeError) as e:
        logger.warning("Prodigy not available (%s). Falling back to AdamW.", e)
        return AdamW(params, lr=lr, weight_decay=weight_decay), None


def warmup_scheduler(optimizer: Any, steps: int) -> Any:
    """The rate from 0 up to the configured one over ``steps`` steps (None when 0)."""
    scheduler = linear_warmup(optimizer, steps, from_zero=True)
    if scheduler is not None:
        logger.info("Configured learning rate warmup for %d steps.", steps)
    return scheduler
