"""The MiniMax H3 LoRA trainer's optimizer: its defaults (AdamW8bit, weight decay 0.01), its Prodigy recipe,
the fall back to AdamW when bitsandbytes or Prodigy cannot be used, and an error for an unknown name."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.optim import AdamW

from maltrainer.optimizers.prodigy import build_prodigy
from maltrainer.training.optimizer import automagic_one_group, eight_bit_or_adamw, linear_warmup

logger = logging.getLogger(__name__)


def build_optimizer(
    params: list[torch.nn.Parameter], train_config: dict[str, Any], lora_dtype: torch.dtype
) -> tuple[Any, dict[str, object] | None]:
    """(optimizer, what it resolved to; None after a fall back) for ``train.optimizer``."""
    lr = float(train_config.get("learning_rate", 1e-4))
    name = str(train_config.get("optimizer", "adamw8bit")).lower()
    weight_decay = float(train_config.get("weight_decay", 0.01))
    if name in {"adamw8bit", "adam8bit"}:
        return eight_bit_or_adamw(params, name, lr, weight_decay)
    if name.startswith("prodigy"):
        return _prodigy(params, train_config, (lr, weight_decay), lora_dtype)
    if name in {"automagic3", "automagic", "automagic_v3"}:
        return automagic_one_group(params, train_config, weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay), {"name": "adamw", "lr": lr}
    raise ValueError(f"Unknown train.optimizer {name!r}: use automagic3, prodigy, adamw8bit, adam8bit or adamw")


def _prodigy(
    params: list[torch.nn.Parameter], train_config: dict[str, Any], rates: tuple[float, float], lora_dtype: torch.dtype
) -> tuple[Any, dict[str, object] | None]:
    lr, weight_decay = rates
    try:
        # Prodigy's `lr` is a multiplier near 1.0: an AdamW-scale learning_rate (<0.01) cannot have meant
        # that, so it falls back to 1.0 there; train.prodigy_lr overrides either way.
        configured_lr = float(train_config.get("learning_rate", 1.0))
        recipe = {  # this trainer's defaults (the shared ones suit the other LoRA trainers)
            "prodigy_beta2": 0.99,
            "prodigy_d_coef": 2.0,  # the most useful knob: ~2.0 is the usual LoRA recommendation
            "prodigy_use_bias_correction": True,
            "prodigy_safeguard_warmup": bool(train_config.get("learning_rate_warmup_steps", 0)),
            "prodigy_slice_p": None,
        }
        optimizer, kwargs = build_prodigy(params, train_config, configured_lr, weight_decay, recipe)
        logger.info("Using Prodigy: %s", ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items())))
        if configured_lr < 0.01 and "prodigy_lr" not in train_config:
            logger.info(
                "train.learning_rate (%.2e) is AdamW-scale, so Prodigy's multiplier is "
                "1.0 instead. Set train.prodigy_lr to control it explicitly.",
                configured_lr,
            )
        if lora_dtype is not torch.float32:
            logger.warning(
                "Prodigy adapts on dot products of gradients and parameter deltas; in "
                "%s those are too coarse to steer it. Set lora.dtype: float32.",
                lora_dtype,
            )
        return optimizer, {"name": "prodigy", **kwargs}
    except (ImportError, OSError, RuntimeError) as e:
        logger.warning("Prodigy not available (%s). Falling back to AdamW.", e)
        return AdamW(params, lr=lr, weight_decay=weight_decay), None


def warmup_scheduler(optimizer: Any, train_config: dict[str, Any]) -> Any:
    """A linear warmup over ``train.learning_rate_warmup_steps`` (None when it is 0)."""
    steps = int(train_config.get("learning_rate_warmup_steps", 0) or 0)
    scheduler = linear_warmup(optimizer, steps)
    if scheduler is not None:
        logger.info("Configured learning rate warmup for %d steps.", steps)
    return scheduler
