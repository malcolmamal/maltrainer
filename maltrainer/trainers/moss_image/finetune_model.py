"""The MOSS-Image finetune's model and optimizer: the plain bf16 DiT with the trainable subset chosen, and fused
Automagic3 over it (restoring its state, after checking it was kept for the same parameters, when resuming)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch

from maltrainer.finetune.resume import check_same_trainable_set
from maltrainer.finetune.selection import select_trainable
from maltrainer.models.flux3.loading import is_int8_convrot
from maltrainer.models.moss_image.finetune import DEFAULT_TRAINABLE_PREFIXES, param_groups
from maltrainer.models.moss_image.loading import load_moss_image_transformer
from maltrainer.optimizers import Automagic3
from maltrainer.trainers.moss_image.finetune_host import STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER

logger = logging.getLogger(__name__)


def load_for_finetune(host: Any, say: Callable[[str], None]) -> None:
    """Sets ``layout_path``, ``transformer``, ``named_trainable`` and ``lora_network`` (the inherited training step
    runs whatever lora_network holds) on the trainer."""
    say("=== Step 2/2: Loading MOSS-Image DiT for finetuning ===")
    train_config = host.config.get("train", {})
    finetune_config = host.config.get("finetune", {}) or {}
    # Saves copy the key layout of model.transformer_path; on resume the weights come from
    # the state folder, which the next save replaces, so it cannot serve as the layout.
    host.layout_path = Path(host.config["model"]["transformer_path"])
    if is_int8_convrot(host.layout_path):
        raise ValueError(
            f"{host.layout_path.name} is int8; finetuning needs the plain bf16 DiT "
            "(moss_image_base_bf16.safetensors, or a model this trainer saved)"
        )
    weights_path = host.layout_path
    if host.resume_dir is not None:
        for name in (STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER):
            if not (host.resume_dir / name).exists():
                raise FileNotFoundError(f"train.resume_from: {host.resume_dir / name} is missing")
        weights_path = host.resume_dir / STATE_MODEL
        logger.info("Resuming from %s", host.resume_dir)

    host.transformer, load_info = load_moss_image_transformer(weights_path, host.device, host.dtype)
    host.run_metadata["base_load"] = load_info
    if bool(train_config.get("gradient_checkpointing", True)):
        host.transformer.enable_gradient_checkpointing()

    # No key = the MOSS-Image default (the blocks); an explicit [] = everything.
    prefixes = finetune_config.get("trainable_prefixes")
    prefixes = list(DEFAULT_TRAINABLE_PREFIXES) if prefixes is None else list(prefixes)
    linears_only = bool(finetune_config.get("linears_only", True))
    host.named_trainable = select_trainable(host.transformer, prefixes, linears_only=linears_only)
    text_side = sum(p.numel() for name, p in host.named_trainable if name.startswith("text_embedder."))
    if text_side:
        logger.warning(
            "finetune.trainable_prefixes trains the text adapter (%.1fM parameters in text_embedder). "
            "OpenMOSS and Fizgig keep it frozen: it reads every prompt, and in Fizgig's runs "
            "training it exaggerated expressions more each epoch.",
            text_side / 1e6,
        )
    host.transformer.train()
    host.lora_network = cast(Any, host.transformer)
    trainable = sum(p.numel() for _, p in host.named_trainable)
    total = sum(p.numel() for p in host.transformer.parameters())
    host.run_metadata.update(
        trainable_parameters=trainable, trainable_prefixes=list(prefixes) or "all", linears_only=linears_only
    )
    logger.info(
        "Finetuning %.3fB of %.3fB parameters (%s%s)",
        trainable / 1e9,
        total / 1e9,
        ", ".join(prefixes) or "all",
        ", Linear weights only" if linears_only else "",
    )
    host._maybe_compile()


def setup_optimizer(host: Any) -> None:
    train_config = host.config["train"]
    name = str(train_config.get("optimizer", "automagic3")).lower()
    if name not in {"automagic3", "automagic", "automagic_v3"}:
        raise ValueError(
            f"MOSS-Image finetuning supports automagic3 only (got {name!r}): it rounds updates "
            "stochastically into the bf16 weights and keeps ~1 byte of state per parameter"
        )
    fused = bool(train_config.get("automagic_fused", True))
    accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)
    if fused and accum > 1:
        raise ValueError("train.automagic_fused steps inside backward, so gradient_accumulation_steps must be 1")
    groups = param_groups(host.named_trainable, bool(train_config.get("automagic_param_groups", True)))
    kwargs: dict[str, Any] = {
        "lr": float(train_config.get("learning_rate", 1e-6)),
        "min_lr": float(train_config.get("automagic_min_lr", 1e-8)),
        "max_lr": float(train_config.get("automagic_max_lr", 1e3)),
        "beta2": float(train_config.get("automagic_beta2", 0.999)),
        "eps": float(train_config.get("automagic_eps", 1e-30)),
        "clip_threshold": float(train_config.get("automagic_clip_threshold", 1.0)),
        "weight_decay": float(train_config.get("weight_decay", 0.0)),
        "polarity_history": int(train_config.get("automagic_polarity_history", 8)),
        "fused": fused,
    }
    host.optimizer = Automagic3(groups, **kwargs)
    host.run_metadata["optimizer_resolved"] = {
        "name": "automagic3",
        "groups": [g["name"] for g in groups],
        **kwargs,
    }
    logger.info("Optimizer: %s", host.run_metadata["optimizer_resolved"])

    if host.resume_dir is not None:
        state = torch.load(host.resume_dir / STATE_TRAINER, map_location="cpu", weights_only=False)
        check_same_trainable_set(state, [g["name"] for g in groups], [name for name, _ in host.named_trainable])
        host.optimizer.load_state_dict(
            torch.load(host.resume_dir / STATE_OPTIMIZER, map_location=host.device, weights_only=False)
        )
        host.resume_state = state
