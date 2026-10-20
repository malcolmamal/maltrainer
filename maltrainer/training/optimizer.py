"""The optimizer over a LoRA's adapters, chosen by ``train.optimizer``, and the learning-rate warmup."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch.optim import AdamW

from maltrainer.optimizers.prodigy import build_prodigy

logger = logging.getLogger(__name__)

AUTOMAGIC_NAMES = {"automagic3", "automagic", "automagic_v3"}


def automagic_kwargs(train_config: dict[str, Any], lr: float, weight_decay: float, history: int) -> dict[str, Any]:
    return {
        "lr": lr,
        "min_lr": float(train_config.get("automagic_min_lr", 1e-8)),
        "max_lr": float(train_config.get("automagic_max_lr", 1e3)),
        "beta2": float(train_config.get("automagic_beta2", 0.999)),
        "eps": float(train_config.get("automagic_eps", 1e-30)),
        "clip_threshold": float(train_config.get("automagic_clip_threshold", 1.0)),
        "weight_decay": weight_decay,
        "polarity_history": int(train_config.get("automagic_polarity_history", history)),
        "fused": bool(train_config.get("automagic_fused", False)),
    }


def build_lora_optimizer(
    network: Any, train_config: dict[str, Any], history: int, strict: bool
) -> tuple[Any, dict[str, object]]:
    """(optimizer, what it resolved to, for the metadata). ``network`` has ``trainable_parameters()`` and
    ``param_groups(split)``; ``history`` is Automagic3's default sign window. An unknown ``train.optimizer``
    is an error when ``strict``, AdamW otherwise."""
    name = str(train_config.get("optimizer", "automagic3")).lower()
    weight_decay = float(train_config.get("weight_decay", 0.0))
    lr = float(train_config.get("learning_rate", 1e-6))
    if name in AUTOMAGIC_NAMES:
        from maltrainer.optimizers import Automagic3

        # Each module family (attention, MLP, I/O ...) adapts its own rate.
        groups = network.param_groups(bool(train_config.get("automagic_param_groups", True)))
        kwargs = automagic_kwargs(train_config, lr, weight_decay, history)
        return Automagic3(groups, **kwargs), {"name": "automagic3", "groups": [g["name"] for g in groups], **kwargs}
    if name.startswith("prodigy"):
        optimizer, supported = build_prodigy(network.trainable_parameters(), train_config, lr, weight_decay)
        return optimizer, {"name": "prodigy", **supported}
    if name in {"adamw8bit", "adam8bit"}:
        import bitsandbytes as bnb

        params = network.trainable_parameters()
        return cast(Any, bnb.optim).AdamW8bit(params, lr=lr, weight_decay=weight_decay), {"name": name, "lr": lr}
    if strict and name != "adamw":
        raise ValueError(f"Unknown train.optimizer {name!r}: use automagic3, prodigy, adamw8bit, adam8bit or adamw")
    return AdamW(network.trainable_parameters(), lr=lr, weight_decay=weight_decay), {"name": "adamw", "lr": lr}


def warmup_scheduler(optimizer: Any, train_config: dict[str, Any]) -> Any:
    """A linear warmup over ``train.learning_rate_warmup_steps`` (None when it is 0)."""
    warmup = int(train_config.get("learning_rate_warmup_steps", 0) or 0)
    if warmup <= 0:
        return None
    logger.info("Learning-rate warmup over %d steps.", warmup)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, 1.0))
