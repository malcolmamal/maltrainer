"""The Qwen-Image 2.1 LoRA trainer's optimizer: its own defaults (AdamW8bit, weight decay 0.01), a fall back
to AdamW when bitsandbytes or Prodigy cannot be used, and its warmup from a zero rate."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.optim import AdamW

from maltrainer.optimizers.prodigy import build_prodigy

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
        return _eight_bit(params, name, lr, weight_decay)
    if name.startswith("prodigy"):
        return _prodigy(params, train_config, lr, weight_decay)
    if name in {"automagic3", "automagic", "automagic_v3"}:
        return _automagic(params, train_config, weight_decay)
    return AdamW(params, lr=lr, weight_decay=weight_decay), {"name": "adamw", "lr": lr}


def _eight_bit(params: list[torch.nn.Parameter], name: str, lr: float, weight_decay: float) -> tuple[Any, Resolved]:
    try:
        import bitsandbytes as bnb

        bnb_optim: Any = bnb.optim
        optimizer_cls = bnb_optim.AdamW8bit if name == "adamw8bit" else bnb_optim.Adam8bit
        optimizer = optimizer_cls(params, lr=lr, weight_decay=weight_decay)
        logger.info("Using 8-bit optimizer: %s (lr=%.2e)", name, lr)
        return optimizer, {"name": name, "lr": lr}
    except (ImportError, OSError, RuntimeError) as e:
        logger.warning("Bitsandbytes not available (%s). Falling back to AdamW.", e)
        return AdamW(params, lr=lr, weight_decay=weight_decay), None


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


def _automagic(
    params: list[torch.nn.Parameter], train_config: dict[str, Any], weight_decay: float
) -> tuple[Any, Resolved]:
    from maltrainer.optimizers import Automagic3

    def option(key: str, default: float) -> Any:  # automagic_<key>, or the bare key
        return train_config.get(f"automagic_{key}", train_config.get(key, default))

    kwargs: dict[str, Any] = {
        "lr": float(train_config.get("learning_rate", 1e-6)),
        "min_lr": float(option("min_lr", 1e-8)),
        "max_lr": float(option("max_lr", 1e3)),
        "beta2": float(option("beta2", 0.999)),
        "eps": float(option("eps", 1e-30)),
        "clip_threshold": float(option("clip_threshold", 1.0)),
        "weight_decay": weight_decay,
        "polarity_history": int(option("polarity_history", 8)),
        "fused": bool(train_config.get("automagic_fused", False)),
    }
    optimizer = Automagic3(params, **kwargs)
    logger.info(
        "Using Automagic3 optimizer: start_lr=%.2e, history=%d, clip_threshold=%.2f",
        kwargs["lr"],
        kwargs["polarity_history"],
        kwargs["clip_threshold"],
    )
    return optimizer, {"name": "automagic3", **kwargs}


def warmup_scheduler(optimizer: Any, steps: int) -> Any:
    """The rate from 0 up to the configured one over ``steps`` steps (None when 0)."""
    if steps <= 0:
        return None
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        return min(1.0, float(current_step) / float(max(1, steps)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    logger.info("Configured learning rate warmup for %d steps.", steps)
    return scheduler
