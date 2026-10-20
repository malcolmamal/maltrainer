"""A training run's set-up: its metadata, the Accelerator, the device report and the training.log file."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator

from maltrainer.utils.lora_metadata import git_commit

logger = logging.getLogger(__name__)


def start_metadata(model_type: str) -> dict[str, object]:
    """What every checkpoint of the run records about how it was made."""
    return {
        "model_type": model_type,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "torch_version": torch.__version__,
    }


def make_accelerator(config: dict[str, Any]) -> Accelerator:
    return Accelerator(
        mixed_precision=config.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=config.get("train", {}).get("gradient_accumulation_steps", 1),
    )


def log_device(device: torch.device) -> None:
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    logger.info("Training on device: %s (%s)", device, name)
    if device.type == "cuda":
        logger.info("Total VRAM: %.2f GB", torch.cuda.get_device_properties(device).total_memory / 1024**3)


def open_training_log(config: dict[str, Any]) -> logging.Logger:
    """<output.path>/training.log gets every log line; returns the ``progress`` logger, whose step lines go
    to that file only."""
    log_dir = Path(config.get("output", {}).get("path", "./output"))
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    progress_logger = logging.getLogger("progress")
    progress_logger.setLevel(logging.INFO)
    progress_logger.addHandler(file_handler)
    progress_logger.propagate = False
    return progress_logger
