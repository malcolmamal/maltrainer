"""The H3 Image finetune's training loop: a step budget over shuffled epochs, the window swapped in every
``finetune.rotate_every`` steps, one fused backward (every trained weight steps inside it), due saves, and an exact
resume (the epoch's order, every RNG state and the likeness epoch come back with the saved position)."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, cast

import torch
from tqdm import tqdm

from maltrainer.trainers.minimax_h3_image.finetune_host import FinetuneHost
from maltrainer.training.steps import plan_training_steps

logger = logging.getLogger(__name__)


def train(host: FinetuneHost) -> None:
    train_config = host.config["train"]
    if not host.samples:
        raise ValueError("No usable images: every file in the dataset failed to load")
    epoch_size = host._epoch_size()
    steps_per_epoch, num_epochs, total_steps = plan_training_steps(epoch_size, train_config)
    host.run_metadata.update(
        steps_per_epoch=steps_per_epoch, planned_total_steps=total_steps, dataset_samples=len(host.samples)
    )
    resume, same_data = host.resume_state, True
    if resume is not None:
        same_data = _resume_position(host, resume, epoch_size)
        if host.global_step >= total_steps:
            logger.warning(
                "The resumed run is already at step %d of %d; raise train.max_steps.", host.global_step, total_steps
            )
            return
        num_epochs = host.epoch + -(-(total_steps - host.global_step + host.position) // steps_per_epoch)
    logger.info(
        "Starting MiniMax H3 Image finetuning: %d epochs, %d total steps, %d windows of %d steps",
        num_epochs,
        total_steps,
        len(host.rotator.windows),
        host._rotate_every(),
    )
    bar = tqdm(total=total_steps, initial=host.global_step, desc="Training Steps", disable=not host.is_main_process)
    _epochs(host, (resume, same_data), (num_epochs, total_steps), bar)
    bar.close()
    host.save_checkpoint("final")
    logger.info("MiniMax H3 Image finetuning completed!")


def _resume_position(host: FinetuneHost, resume: dict[str, Any], epoch_size: int) -> bool:
    """Where the saved run stands; whether it was on this dataset (else the next epoch starts fresh)."""
    host.global_step, host.epoch, host.position = resume["global_step"], resume["epoch"], resume["position"]
    host._steps_taken = resume["steps_taken"]
    if resume.get("dataset_fingerprint") == host._dataset_fingerprint():
        logger.info(
            "Resumed at step %d (epoch %d, sample %d of its order)", host.global_step, host.epoch + 1, host.position
        )
        return True
    host.epoch, host.position = host.epoch + (1 if host.position > 0 else 0), 0
    logger.warning(
        "The dataset changed since the saved state (%s samples then, %d now); resuming at step %d with a fresh epoch.",
        resume.get("dataset_samples", "?"),
        epoch_size,
        host.global_step,
    )
    return False


def _restore_rngs(resume: dict[str, Any]) -> None:
    random.setstate(resume["python_rng"])
    torch.set_rng_state(resume["torch_rng"])
    if resume.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(resume["cuda_rng"])


def _epochs(host: FinetuneHost, resumed: tuple[dict[str, Any] | None, bool], budget: tuple[int, int], bar: Any) -> None:
    """Epochs until the step budget is spent."""
    resume, same_data = resumed
    num_epochs, total_steps = budget
    epoch, first = host.epoch, resume is not None
    while host.global_step < total_steps:
        num_epochs = max(num_epochs, epoch + 1)
        if resume is not None and first and same_data:
            random.setstate(resume["epoch_rng"])
        host.epoch, host.epoch_rng = epoch, random.getstate()
        order = host._epoch_order()
        start = host.position if first else 0
        if resume is not None and first:
            _restore_rngs(resume)
        first = False
        for pos in range(start, len(order)):
            _sample_step(host, order[pos], (pos, epoch, num_epochs, total_steps), bar)
            if host.global_step >= total_steps:
                break
        if host.global_step < total_steps:
            host.position = 0
            epoch += 1


def _activate(host: FinetuneHost) -> int:
    """The window for this step, swapped in when it changes."""
    window = host._window_at(host.global_step)
    if window != host.rotator.current:
        started = time.time()
        host.rotator.activate(window)
        host.optimizer = host.rotator.optimizer  # what the control API's stats read
        logger.info(
            "Step %d: training window %s (%.2fB, %.1fs to swap)",
            host.global_step,
            host.rotator.names[window],
            host.rotator.window_parameters(window) / 1e9,
            time.time() - started,
        )
    return window


def _sample_step(host: FinetuneHost, index: int, where: tuple[int, int, int, int], bar: Any) -> None:
    pos, epoch, num_epochs, total_steps = where
    window = _activate(host)
    sample = host.samples[index]
    loss, parts = host._training_step(sample, 1.0)
    cast(Any, loss).backward()  # fused: every trained weight steps here
    host.rotator.optimizer.step()  # applies the window's rate votes
    if host.always_optimizer is not None:
        host.always_optimizer.step()
    host.global_step += 1
    host.position = pos + 1
    bar.update(1)
    bar.set_postfix(
        {"loss": f"{loss.item():.4f}", "window": host.rotator.names[window], "epoch": f"{epoch + 1}/{num_epochs}"}
    )
    host.progress_logger.info(
        "Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, Window: %s, LR: %.2e, Sigma: %.3f",
        host.global_step,
        total_steps,
        int(host.global_step / total_steps * 100),
        loss.item(),
        epoch + 1,
        num_epochs,
        host.rotator.names[window],
        host.rotator.optimizer.get_avg_learning_rate(),
        parts["sigma"],
    )
    save_every = int(host.config["train"].get("save_every", 0) or 0)
    if save_every > 0 and host.global_step % save_every == 0 and host.global_step < total_steps:
        host.save_checkpoint(host.global_step)
