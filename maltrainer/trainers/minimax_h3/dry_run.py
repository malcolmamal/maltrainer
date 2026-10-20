"""The MiniMax H3 LoRA trainer's dry run: the step count from the dataset's files, then simulated steps that
write a JSON progress record where each checkpoint would go (the web UI follows them)."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


def simulate(config: dict[str, Any], num_samples: int, step_time: float, show: bool) -> str:
    """Simulate training without downloading or loading models; returns the last record's path."""
    train_config = config.get("train", {})
    num_epochs = int(train_config.get("num_epochs", 10) or 1)
    batch_size = int(train_config.get("batch_size", 1) or 1)
    grad_accum = int(train_config.get("gradient_accumulation_steps", 1) or 1)
    steps_per_epoch = math.ceil((num_samples / max(batch_size, 1)) / max(grad_accum, 1))
    total_steps = max(num_epochs * steps_per_epoch, 1)
    save_every = int(train_config.get("save_every", 500) or 0)
    step_time = max(float(step_time), 0.0)
    output_dir = Path(config.get("output", {}).get("path", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dry run enabled: simulating %d steps.", total_steps)

    plan = (total_steps, steps_per_epoch, num_epochs, time.time())
    progress_bar = tqdm(total=total_steps, desc="Dry Run Steps", disable=not show)
    for step in range(1, total_steps + 1):
        if step_time > 0:
            time.sleep(step_time)
        progress_bar.update(1)
        if save_every > 0 and step % save_every == 0:
            _write_record(config, output_dir, step, plan)
    progress_bar.close()
    last = _write_record(config, output_dir, "final", plan)
    logger.info("Dry run completed successfully.")
    return last


def _write_record(config: dict[str, Any], output_dir: Path, step: Any, plan: tuple[int, int, int, float]) -> str:
    total_steps, steps_per_epoch, num_epochs, start_time = plan
    elapsed = max(time.time() - start_time, 0.0)
    completed_steps = total_steps if step == "final" else int(step)
    progress = min(completed_steps / max(total_steps, 1), 1.0)
    payload = {
        "mode": "dry_run",
        "kind": "minimax_h3_lora",
        "step": step,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "num_epochs": num_epochs,
        "progress": {
            "completed_steps": completed_steps,
            "percent": round(progress * 100, 2),
            "elapsed_seconds": round(elapsed, 4),
        },
        "config": config,
    }
    save_path = output_dir / f"minimax_h3_lora_step_{step}.safetensors"
    save_path.write_text(json.dumps(payload, indent=2))
    return str(save_path)
