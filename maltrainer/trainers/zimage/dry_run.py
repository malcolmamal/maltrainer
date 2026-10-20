"""The Z-Image trainers' dry run: the step count from the dataset's files, then simulated steps that write a
JSON progress record where each checkpoint would go (the web UI follows them)."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


def grad_accum_steps(config: dict[str, Any]) -> Any:
    """train.gradient_accumulation_steps, else the top-level one (where the accelerator reads it)."""
    return config.get("train", {}).get("gradient_accumulation_steps", config.get("gradient_accumulation_steps", 1))


def simulate(config: dict[str, Any], num_samples: int, step_time: float, kind: str, show: bool) -> str:
    """Simulate training without loading or downloading models; returns the last record's path. ``kind``: the
    records' "kind" and file-name prefix ("lora" -> zimage_lora_step_<N>.safetensors)."""
    train_config = config.get("train", {})
    num_epochs = int(train_config.get("num_epochs", 10) or 1)
    batch_size = int(train_config.get("batch_size", 1) or 1)
    grad_accum = int(grad_accum_steps(config) or 1)
    steps_per_epoch = math.ceil((num_samples / max(batch_size, 1)) / max(grad_accum, 1))
    total_steps = max(num_epochs * steps_per_epoch, 1)
    save_every = int(train_config.get("save_every", 500) or 0)
    raw_step_time, step_time = float(step_time), max(float(step_time), 0.0)
    output_dir = Path(config.get("output", {}).get("path", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dry run enabled: skipping model download/loading and training compute.")
    logger.info(
        "Simulating %d steps (epochs=%d, steps/epoch=%d, batch_size=%d, grad_accum=%d, step_time=%.2fs).",
        total_steps,
        num_epochs,
        steps_per_epoch,
        batch_size,
        grad_accum,
        step_time,
    )
    run = (kind, total_steps, steps_per_epoch, num_epochs, time.time(), raw_step_time)
    progress_bar = tqdm(total=total_steps, desc="Dry Run Steps", disable=not show)
    for step in range(1, total_steps + 1):
        if step_time > 0:
            time.sleep(step_time)
        progress_bar.update(1)
        if save_every > 0 and step % save_every == 0:
            _write_record(config, output_dir, step, run)
    progress_bar.close()
    last = _write_record(config, output_dir, "final", run)
    logger.info("Dry run completed.")
    return last


def _write_record(
    config: dict[str, Any], output_dir: Path, step: Any, run: tuple[str, int, int, int, float, float]
) -> str:
    kind, total_steps, steps_per_epoch, num_epochs, start_time, step_time = run
    dataset_config, train_config = config.get("dataset", {}), config.get("train", {})
    elapsed = max(time.time() - start_time, 0.0)
    completed_steps = total_steps if step == "final" else int(step)
    progress = min(completed_steps / max(total_steps, 1), 1.0)
    payload = {
        "mode": "dry_run",
        "kind": kind,
        "step": step,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "num_epochs": num_epochs,
        "progress": {
            "completed_steps": completed_steps,
            "percent": round(progress * 100, 2),
            "elapsed_seconds": round(elapsed, 4),
            "eta_seconds": round(max((total_steps - completed_steps) * step_time, 0.0), 4),
            "step_time_seconds": step_time,
        },
        "config": config,
        "derived": {
            "dataset_path": str(dataset_config.get("path", "")),
            "dataset_resolution": dataset_config.get("resolution", 512),
            "batch_size": train_config.get("batch_size", 1),
            "gradient_accumulation_steps": grad_accum_steps(config),
            "save_every": train_config.get("save_every", 500),
            "dtype": str(config.get("dtype", "bfloat16")),
            "mixed_precision": str(config.get("mixed_precision", "bf16")),
            "output_path": str(output_dir),
        },
    }
    save_path = output_dir / f"zimage_{kind}_step_{step}.safetensors"
    save_path.write_text(json.dumps(payload, indent=2))
    return str(save_path)
