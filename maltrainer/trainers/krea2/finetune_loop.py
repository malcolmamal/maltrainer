"""The Krea 2 finetune's training loop: a step budget over shuffled epochs (per-folder repeats and regularisation
images included), the prior recorded before the first step, the base model's previews, keys, and an exact resume
(the epoch's order and every RNG state come back with the saved position). One step is finetune_step.py: the
window swapped in every ``finetune.rotate_every`` steps, the prior term, fused Automagic3 inside backward."""

from __future__ import annotations

import logging
import random
from typing import Any

import torch
from tqdm import tqdm

from maltrainer.control.launch import make_preview_host_keys
from maltrainer.trainers.krea2.finetune_host import FinetuneHost
from maltrainer.trainers.krea2.finetune_step import sample_step
from maltrainer.training.steps import plan_training_steps
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.train_monitor import TrainMonitor

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
    host._setup_prior()
    if host.reg_keys:
        host.run_metadata.update({"reg_images": len(host.reg_keys), "reg_lr_multiplier": host.reg_multiplier})

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
    monitor = TrainMonitor(every=int(train_config.get("monitor_every", 100) or 100))
    keys = None
    if host.is_main_process:
        gallery = host._run_gallery if host.gallery.enabled else None
        keys = make_preview_host_keys(KeyCommands, host, monitor, gallery, kind="finetune")
    if resume is None:
        monitor.exclude(host._maybe_preview(0, total_steps))  # the base model: what later previews compare with
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
            stopped = sample_step(host, order[pos], (pos, epoch, num_epochs, total_steps), tools, shift_logged)
            if host.global_step >= total_steps or stopped:
                break
        if stopped:
            return True
        if host.global_step < total_steps:
            host.position = 0
            epoch += 1
    return False
