"""The Z-Image trainers' optimizer: AdamW (the default), Prodigy or bitsandbytes' 8-bit Adam, from settings
that may be YAML strings ("1e-4", "yes"); a missing Prodigy or bitsandbytes falls back to AdamW."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import AdamW

logger = logging.getLogger(__name__)


def to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def to_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value_str = str(value).strip().lower()
    if value_str in {"true", "1", "yes", "y", "on"}:
        return True
    if value_str in {"false", "0", "no", "n", "off"}:
        return False
    return bool(default)


def build_optimizer(
    params: Iterable[torch.nn.Parameter], train_config: dict[str, Any], default_lr: float, lora: bool = True
) -> torch.optim.Optimizer:
    """``lora``: the LoRA trainer's extras, the 8-bit optimizers and ``weight_decay_exclude_lora`` (the embedding
    trainer has neither: an unknown or 8-bit name is plain AdamW there)."""
    params = list(params)  # a fall back after a failed Prodigy gets them all again
    lr = to_float(train_config.get("learning_rate", default_lr), default_lr)
    betas = (to_float(train_config.get("adam_beta1", 0.9), 0.9), to_float(train_config.get("adam_beta2", 0.999), 0.999))
    weight_decay = to_float(train_config.get("weight_decay", 0.01), 0.01)
    eps = to_float(train_config.get("adam_epsilon", 1e-8), 1e-8)
    optimizer_name = str(train_config.get("optimizer", "adamw")).lower()
    use_weight_decay = 0.0 if lora and train_config.get("weight_decay_exclude_lora", False) else weight_decay
    adam: dict[str, Any] = {"lr": lr, "betas": betas, "weight_decay": use_weight_decay, "eps": eps}

    if optimizer_name.startswith("prodigy"):
        try:
            return _prodigy(params, train_config, adam, optimizer_name)
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning("Failed to load prodigy optimizer (%s). Falling back to AdamW. Error: %s", optimizer_name, e)
            return AdamW(params, **adam)
    if lora and optimizer_name in {"adamw8bit", "adam8bit"}:
        try:
            import bitsandbytes as bnb

            bnb_optim: Any = bnb.optim
            optimizer_cls = bnb_optim.AdamW8bit if optimizer_name == "adamw8bit" else bnb_optim.Adam8bit
            optimizer: torch.optim.Optimizer = optimizer_cls(params, **adam)
            logger.info("Using optimizer: %s", optimizer_name)
            return optimizer
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning(
                "Failed to load bitsandbytes optimizer (%s). Falling back to AdamW. Error: %s", optimizer_name, e
            )
            return AdamW(params, **adam)
    return AdamW(params, **adam)


def _prodigy(
    params: Iterable[torch.nn.Parameter], train_config: dict[str, Any], adam: dict[str, Any], name: str
) -> torch.optim.Optimizer:
    from prodigyopt import Prodigy

    beta3_raw = train_config.get("prodigy_beta3")
    optimizer: torch.optim.Optimizer = Prodigy(
        params,
        **adam,
        beta3=None if beta3_raw is None else to_float(beta3_raw, 0.0),
        decouple=to_bool(train_config.get("prodigy_decouple", True), True),
        use_bias_correction=to_bool(train_config.get("prodigy_use_bias_correction", False), False),
        safeguard_warmup=to_bool(train_config.get("prodigy_safeguard_warmup", False), False),
        d0=to_float(train_config.get("prodigy_d0", 1e-6), 1e-6),
        d_coef=to_float(train_config.get("prodigy_d_coef", 1.0), 1.0),
        growth_rate=to_float(train_config.get("prodigy_growth_rate", float("inf")), float("inf")),
        slice_p=to_int(train_config.get("prodigy_slice_p", 1), 1),
    )
    logger.info("Using optimizer: %s", name)
    return optimizer
