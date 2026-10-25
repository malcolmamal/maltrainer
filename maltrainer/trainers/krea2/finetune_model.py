"""The Krea 2 finetune's model: the DiT with its master copy on the CPU, the window rotator over it, and the
always-trained text-fusion linears; from a resume state's weights when resuming."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import torch

from maltrainer.finetune.rotation import WindowRotator
from maltrainer.models.krea2.finetune import (
    DEFAULT_ALWAYS_ON,
    always_on_linears,
    dequantize_always_on,
    parse_windows,
)
from maltrainer.models.krea2.loading import load_krea2_transformer
from maltrainer.trainers.krea2.finetune_host import STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER, FinetuneHost

logger = logging.getLogger(__name__)


def load_for_finetune(host: FinetuneHost, say: Callable[[str], None]) -> None:
    """Sets ``transformer``, ``key_prefix``, ``rotator``, ``always_on`` and ``lora_network`` (the inherited
    training step calls whatever lora_network holds) on the trainer."""
    say("=== Step 2/2: Loading Krea 2 DiT for finetuning ===")
    model_config = host.config["model"]
    finetune_config = host.config.get("finetune", {}) or {}
    path = Path(model_config["transformer_path"])
    if host.resume_dir is not None:
        for name in (STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER):
            if not (host.resume_dir / name).exists():
                raise FileNotFoundError(f"train.resume_from: {host.resume_dir / name} is missing")
        path = host.resume_dir / STATE_MODEL
        logger.info("Resuming from %s", host.resume_dir)
    master: dict[str, torch.Tensor] = {}
    host.transformer, info = load_krea2_transformer(path, host.device, host.dtype, master=master)
    host.key_prefix = str(info.get("key_prefix", ""))
    host.run_metadata["base_load"] = {k: v for k, v in info.items() if k != "key_prefix"}
    if info.get("source_fp8_layers") and host.resume_dir is None:
        logger.warning(
            "%s stores its blocks in fp8: the finetune starts from dequantized fp8 weights, so every block "
            "keeps fp8's rounding. Use the bf16 Raw model (krea2_raw_bf16.safetensors) for a real run.",
            path.name,
        )
    host.transformer.enable_gradient_checkpointing()
    host.transformer.train()

    host.rotator = WindowRotator(
        host.transformer,
        master,
        parse_windows(finetune_config.get("windows")),
        host.device,
        host.dtype,
        host._make_optimizer,
    )
    always = finetune_config.get("always_on", list(DEFAULT_ALWAYS_ON)) or []
    dequantized = dequantize_always_on(host.transformer, always, host.rotator.master)
    host.always_on = always_on_linears(host.transformer, always)
    if dequantized:
        logger.warning(
            "%d always-trained Linears are stored in fp8: dequantized to trainable %s (they keep fp8's rounding)",
            dequantized,
            str(host.dtype).replace("torch.", ""),
        )
    host.lora_network = host.transformer
    windows = ", ".join(f"{n} {host.rotator.window_parameters(i) / 1e9:.2f}B" for i, n in enumerate(host.rotator.names))
    always_count = sum(p.numel() for _, p in host.always_on)
    host.run_metadata.update(
        {
            "windows": host.rotator.names,
            "always_on": list(always),
            "rotate_every": host._rotate_every(),
            "rotating_parameters": sum(host.rotator.window_parameters(i) for i in range(len(host.rotator.windows))),
            "always_on_parameters": always_count,
        }
    )
    logger.info(
        "Windows: %s; always trained: %s (%.2fB); rotating every %d steps",
        windows,
        ", ".join(always) or "nothing",
        always_count / 1e9,
        host._rotate_every(),
    )
