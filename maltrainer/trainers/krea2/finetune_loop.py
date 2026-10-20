"""The Krea 2 finetune's training loop: a step budget over shuffled epochs, a window of blocks swapped in every
``finetune.rotate_every`` steps, fused Automagic3 stepping inside backward, saves and keys at the step
boundaries, and an exact resume (the epoch's order and every RNG state come back with the saved position)."""

from __future__ import annotations

import logging
import random
import time
from typing import Any, cast

import torch
from tqdm import tqdm

from maltrainer.trainers.krea2.finetune_host import FinetuneHost
from maltrainer.training.steps import plan_training_steps
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.train_monitor import TrainMonitor

logger = logging.getLogger(__name__)


def train(host: FinetuneHost) -> None:
    train_config = host.config["train"]
    if not host.samples:
        raise ValueError("No usable images: every file in the dataset failed to load")
    epoch_size = len(host._epoch_order())
    steps_per_epoch, num_epochs, total_steps = plan_training_steps(epoch_size, train_config)
    host.run_metadata.update(
        steps_per_epoch=steps_per_epoch, planned_total_steps=total_steps, dataset_samples=len(host.samples)
    )

    resume = host.resume_state
    same_data = True
    if resume is not None:
        same_data = _resume_position(host, resume, epoch_size)
        if host.global_step >= total_steps:
            logger.warning(
                "The resumed run is already at step %d of %d; raise train.max_steps to continue.",
                host.global_step,
                total_steps,
            )
            return
        num_epochs = host.epoch + -(-(total_steps - host.global_step + host.position) // steps_per_epoch)

    logger.info(
        "Starting Krea 2 finetuning: %d epochs, %d total steps, %d windows of %d steps",
        num_epochs,
        total_steps,
        len(host.rotator.windows),
        host._rotate_every(),
    )
    progress_bar = tqdm(
        total=total_steps, initial=host.global_step, desc="Training Steps", disable=not host.is_main_process
    )
    keys = KeyCommands() if host.is_main_process else None
    monitor = TrainMonitor(every=int(train_config.get("monitor_every", 100) or 100))
    stopped = _epochs(host, (resume, same_data), (num_epochs, total_steps), (progress_bar, keys, monitor))

    progress_bar.close()
    if keys is not None:
        keys.close()
    if stopped:
        logger.info("Krea 2 finetuning stopped by key at step %d; resume from the _state folder.", host.global_step)
        return
    host.save_checkpoint("final")
    logger.info("Krea 2 finetuning completed!")


def _resume_position(host: FinetuneHost, resume: dict[str, Any], epoch_size: int) -> bool:
    """Where the saved run stands; whether it was on this dataset (else the next epoch starts fresh)."""
    host.global_step, host.epoch, host.position = resume["global_step"], resume["epoch"], resume["position"]
    same_data = resume.get("dataset_fingerprint") == host._dataset_fingerprint()
    if same_data:
        logger.info(
            "Resumed at step %d (epoch %d, sample %d of its order)", host.global_step, host.epoch + 1, host.position
        )
        return True
    if host.position > 0:
        host.epoch += 1
    host.position = 0
    logger.warning(
        "The dataset changed since the saved state (%s samples then, %d now); resuming at step %d with a fresh epoch.",
        resume.get("dataset_samples", "?"),
        epoch_size,
        host.global_step,
    )
    return False


def _epochs(
    host: FinetuneHost,
    resumed: tuple[dict[str, Any] | None, bool],
    budget: tuple[int, int],
    tools: tuple[Any, KeyCommands | None, TrainMonitor],
) -> bool:
    """Epochs until the step budget is spent or a key stops the run; True when a key did."""
    resume, same_data = resumed
    num_epochs, total_steps = budget
    shift_logged = [False]
    epoch = host.epoch
    first = resume is not None
    while host.global_step < total_steps:
        num_epochs = max(num_epochs, epoch + 1)
        if resume is not None and first and same_data:
            random.setstate(resume["epoch_rng"])
        host.epoch, host.epoch_rng = epoch, random.getstate()
        order = host._epoch_order()
        start = host.position if first else 0
        if resume is not None and first:
            random.setstate(resume["python_rng"])
            torch.set_rng_state(resume["torch_rng"])
            if resume.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(resume["cuda_rng"])
        first = False
        stopped = False
        for pos in range(start, len(order)):
            stopped = _sample(host, order[pos], (pos, epoch, num_epochs, total_steps), tools, shift_logged)
            if host.global_step >= total_steps or stopped:
                break
        if stopped:
            return True
        if host.global_step < total_steps:
            host.position = 0
            epoch += 1
    return False


def _sample(
    host: FinetuneHost,
    index: int,
    where: tuple[int, int, int, int],
    tools: tuple[Any, KeyCommands | None, TrainMonitor],
    shift_logged: list[bool],
) -> bool:
    """One sample: the window for this step, loss and fused backward, the log line, a due save, the keys."""
    pos, epoch, num_epochs, total_steps = where
    progress_bar, keys, monitor = tools
    window = host._window_at(host.global_step)
    if window != host.rotator.current:
        started = time.time()
        host.rotator.activate(window)
        monitor.exclude(time.time() - started)
        logger.info(
            "Step %d: training window %s (%.2fB, %.1fs to swap)",
            host.global_step,
            host.rotator.names[window],
            host.rotator.window_parameters(window) / 1e9,
            time.time() - started,
        )
    loss, _, sigma = host._training_step(index, shift_logged)
    cast(Any, loss).backward()  # fused: every trained weight steps here
    host.rotator.optimizer.step()  # applies the window's rate votes
    if host.always_optimizer is not None:
        host.always_optimizer.step()
    monitor.record(loss.item(), float(sigma))
    host.global_step += 1
    host.position = pos + 1
    progress_bar.update(1)
    progress_bar.set_postfix(
        {"loss": f"{loss.item():.4f}", "window": host.rotator.names[window], "epoch": f"{epoch + 1}/{num_epochs}"}
    )
    host._log_step(host.global_step, total_steps, loss, epoch, num_epochs)
    monitor.step_done(host.global_step, total_steps, host.rotator.optimizer)
    save_every = int(host.config["train"].get("save_every", 0) or 0)
    if save_every > 0 and host.global_step % save_every == 0 and host.global_step < total_steps:
        host.save_checkpoint(host.global_step)
    stopped = keys is not None and keys.handle(host.global_step, host.save_checkpoint)
    if keys is not None:
        monitor.exclude(keys.take_paused_seconds())
    return stopped
