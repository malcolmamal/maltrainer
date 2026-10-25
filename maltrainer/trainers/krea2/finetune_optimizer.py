"""The Krea 2 finetune's optimizer: fused Automagic3 only (it rounds updates stochastically into the bf16 weights and
steps inside backward), one per window (made by the rotator) and one for the always-trained text fusion; on resume,
their saved states and the trainer counters."""

from __future__ import annotations

import logging
from typing import Any

import torch

from maltrainer.trainers.krea2.finetune_host import STATE_OPTIMIZER, STATE_TRAINER, FinetuneHost

logger = logging.getLogger(__name__)

AUTOMAGIC_NAMES = {"automagic3", "automagic", "automagic_v3"}


def automagic_kwargs(train_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "lr": float(train_config.get("learning_rate", 1e-5)),
        "min_lr": float(train_config.get("automagic_min_lr", 1e-5)),
        "max_lr": float(train_config.get("automagic_max_lr", 1e-5)),
        "beta2": float(train_config.get("automagic_beta2", 0.999)),
        "eps": float(train_config.get("automagic_eps", 1e-30)),
        "clip_threshold": float(train_config.get("automagic_clip_threshold", 1.0)),
        "weight_decay": float(train_config.get("weight_decay", 0.0)),
        "polarity_history": int(train_config.get("automagic_polarity_history", 8)),
        "fused": True,
    }


def setup_optimizer(host: FinetuneHost) -> None:
    """Sets ``always_optimizer`` and, on resume, every window's optimizer state and ``resume_state``."""
    name = str(host.config["train"].get("optimizer", "automagic3")).lower()
    if name not in AUTOMAGIC_NAMES:
        raise ValueError(
            f"Krea 2 finetuning supports automagic3 only (got {name!r}): it rounds updates stochastically "
            "into the bf16 weights and steps inside backward"
        )
    host.run_metadata["optimizer_resolved"] = {"name": "automagic3", **automagic_kwargs(host.config["train"])}
    logger.info("Optimizer: %s", host.run_metadata["optimizer_resolved"])
    if host.always_on:
        host.always_optimizer = host._make_optimizer([p for _, p in host.always_on], "txtfusion")
    if host.resume_dir is not None:
        saved = torch.load(host.resume_dir / STATE_OPTIMIZER, map_location="cpu", weights_only=False)
        host.rotator.load_state_dict(saved["rotation"])
        if host.always_optimizer is not None and saved.get("always") is not None:
            host.always_optimizer.load_state_dict(saved["always"])
        host.resume_state = torch.load(host.resume_dir / STATE_TRAINER, map_location="cpu", weights_only=False)
