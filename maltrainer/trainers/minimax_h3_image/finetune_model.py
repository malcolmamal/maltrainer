"""The H3 Image finetune's model: the INT8 ConvRot transformer, the window rotator over its block Linears, the
always-trained modules, the frozen training adapter when it is switched on, and a resumed run's trained weights."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from maltrainer.finetune.rotation_int8 import Int8WindowRotator
from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRANetwork
from maltrainer.models.minimax_h3.lora_assistant import attach_assistant
from maltrainer.models.minimax_h3_image.finetune import always_on_params, plan_windows
from maltrainer.models.minimax_h3_image.finetune_int8 import make_swappable
from maltrainer.models.minimax_h3_image.finetune_save import read_resume_weights
from maltrainer.models.minimax_h3_image.grid import file_pdd_sigmas
from maltrainer.models.minimax_h3_image.model import MiniMaxH3ImageTransformer
from maltrainer.trainers.minimax_h3.model import load_transformer
from maltrainer.trainers.minimax_h3_image.finetune_host import (
    STATE_OPTIMIZER,
    STATE_TRAINER,
    STATE_WEIGHTS,
    FinetuneHost,
)

logger = logging.getLogger(__name__)


def attach_training_adapter(host: FinetuneHost, say: Callable[[str], None]) -> None:
    """Fizgig's frozen training adapter over the Linears, unless ``finetune.training_adapter`` is false.

    As in the LoRA recipe it shapes what the weights are trained against and is not in the saved model, so the
    finetuned model runs without it."""
    model_config, finetune_config = host.config["model"], host.config.get("finetune", {}) or {}
    path = model_config.get("training_adapter_path")
    switched_on = bool(finetune_config.get("training_adapter", True))
    host.adapter_layers = nn.ModuleDict()
    host.run_metadata["training_adapter_on"] = bool(path) and switched_on
    if not path:
        return
    if not switched_on:
        say("Training adapter: off (finetune.training_adapter is false)")
        return
    options = (
        str(host.config.get("lora", {}).get("prefix", "diffusion_model.")),
        float(finetune_config.get("training_adapter_multiplier", 1.0)),
        model_config.get("training_adapter_adaln_grid"),
    )
    layers, attached = attach_assistant(
        host.transformer, path, options, (host.device, host.dtype), MiniMaxH3LoRANetwork._wrap_forward, logger
    )
    host.adapter_layers.update(layers)
    host.run_metadata.update(training_adapter=Path(str(path)).name, training_adapter_modules=attached)


def _restore(host: FinetuneHost, resume_dir: Path) -> None:
    """The weights a resumed run had trained: the blocks' into the rotator, the always-trained ones in place."""
    trained, always = read_resume_weights(resume_dir / STATE_WEIGHTS)
    names = [name for name, _ in host.always_on]
    if sorted(always) != sorted(names):
        raise ValueError(
            f"The resumed run trained {sorted(always)} beside its windows, this config trains {sorted(names)}; "
            "keep finetune.always_on as it was"
        )
    host.rotator.restore(trained)
    with torch.no_grad():
        for name, param in host.always_on:
            param.copy_(always[name].to(param.device, param.dtype))
    logger.info("Resumed %d trained block weights and %d always-trained ones", len(trained), len(always))


def load_for_finetune(host: FinetuneHost, say: Callable[[str], None]) -> None:
    """Sets ``transformer``, ``lora_network``, ``sigmas``, ``rotator``, ``always_on`` and ``adapter_layers``."""
    say("=== Step 2/2: Loading MiniMax H3 Image INT8 Transformer for finetuning ===")
    finetune_config = host.config.get("finetune", {}) or {}
    if host.resume_dir is not None:
        for name in (STATE_WEIGHTS, STATE_OPTIMIZER, STATE_TRAINER):
            if not (host.resume_dir / name).exists():
                raise FileNotFoundError(f"train.resume_from: {host.resume_dir / name} is missing")
        logger.info("Resuming from %s", host.resume_dir)
    sigmas = file_pdd_sigmas(host.config["model"]["transformer_path"])
    transformer = load_transformer(
        host.config, host.device, host.run_metadata, say, lambda c: MiniMaxH3ImageTransformer(c, len(sigmas) - 1)
    )
    host.transformer = host.lora_network = transformer
    host.sigmas = sigmas
    host.run_metadata["noise_levels"] = list(sigmas[:-1])
    keys = make_swappable(transformer)  # before the adapter wraps the forwards
    attach_training_adapter(host, say)
    windows = plan_windows(keys, finetune_config)
    host.rotator = Int8WindowRotator(transformer, windows, host.device, host.dtype, host._make_optimizer)
    always = list(finetune_config.get("always_on") or [])
    host.always_on = always_on_params(transformer, always)
    if host.resume_dir is not None:
        _restore(host, host.resume_dir)
    sizes = ", ".join(f"{n} {host.rotator.window_parameters(i) / 1e9:.2f}B" for i, n in enumerate(host.rotator.names))
    host.run_metadata.update(
        windows=host.rotator.names,
        always_on=always,
        rotate_every=host._rotate_every(),
        rotating_parameters=sum(host.rotator.window_parameters(i) for i in range(len(windows))),
        always_on_parameters=sum(p.numel() for _, p in host.always_on),
    )
    logger.info(
        "Windows: %s; always trained: %s; rotating every %d steps", sizes, always or "nothing", host._rotate_every()
    )
