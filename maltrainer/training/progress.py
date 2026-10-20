"""The per-step progress line in training.log (the web UI follows it), and the dry run's simulated steps."""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def log_step(
    progress_logger: logging.Logger,
    optimizer: Any,
    step: int,
    total: int,
    loss: torch.Tensor,
    epoch: int,
    num_epochs: int,
    note: str = "",
) -> None:
    """``Step: n/total [p%], Loss, Epoch, <note>LR``, the rate as the optimizer has it: Automagic3's average
    and per-group rates, Prodigy's multiplier x d, or the plain rate."""
    group = optimizer.param_groups[0]
    inner = getattr(optimizer, "optimizer", optimizer)  # accelerate's wrapper does not forward custom methods
    prefix = (
        f"Step: {step}/{total} [{int(step / total * 100)}%], Loss: {loss.item():.6f}, Epoch: {epoch + 1}/{num_epochs}, "
    )
    prefix += note
    if hasattr(inner, "get_avg_learning_rate"):
        rates = ", ".join(
            f"{g.get('name', i)} {lr:.2e}"
            for i, (g, lr) in enumerate(zip(inner.param_groups, inner.get_learning_rates(), strict=False))
        )
        progress_logger.info("%sLR: %.2e (automagic3: %s)", prefix, inner.get_avg_learning_rate(), rates)
    elif group.get("d") is not None:
        progress_logger.info(
            "%sLR: %.2e (mult %.3f x d %.2e)", prefix, group["lr"] * group["d"], group["lr"], group["d"]
        )
    else:
        progress_logger.info("%sLR: %.2e", prefix, group["lr"])


def simulate_steps(
    progress_logger: logging.Logger,
    plan: tuple[int, int, int],
    step_time: float,
    show: bool,
    save_every: int | None = None,
) -> None:
    """Tick through ``plan`` (steps_per_epoch, num_epochs, total_steps) as a real run would log it, without models.
    ``save_every``: also log where a checkpoint would be saved."""
    steps_per_epoch, num_epochs, total_steps = plan
    with tqdm(total=total_steps, desc="Training Steps (dry run)", disable=not show) as bar:
        for step in range(1, total_steps + 1):
            time.sleep(step_time)
            bar.update(1)
            progress_logger.info(
                "Step: %d/%d [%d%%], Loss: 0.000000, Epoch: %d/%d, LR: 0.00e+00 (dry run)",
                step,
                total_steps,
                int(step / total_steps * 100),
                math.ceil(step / steps_per_epoch),
                num_epochs,
            )
            if save_every is not None and step % save_every == 0 and step < total_steps:
                logger.info("Dry run: would save checkpoint at step %d", step)
    logger.info("Dry run complete: would save final checkpoint at step %d", total_steps)
