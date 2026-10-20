"""The MOSS-Image finetune's training loop: a step budget over shuffled epochs, prior preservation and
regularisation images folded into each step, fused Automagic3 stepping inside backward, EMA, previews, the
monitor, saves and keys at the step boundaries, and an exact resume (the epoch's order and every RNG state come
back with the saved position)."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from tqdm import tqdm

from maltrainer.finetune.regularisation import set_step_scale
from maltrainer.trainers.moss_image.finetune_resume import accumulation, build_monitor, resume_position, start_ema
from maltrainer.training.steps import plan_training_steps
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.train_monitor import TrainMonitor

logger = logging.getLogger(__name__)


@dataclass
class Accumulation:
    """What builds up between optimizer steps: micro-steps, their regularisation scales and prior losses."""

    micro: int = 0
    scales: list[float] = field(default_factory=list)
    prior_losses: list[float] = field(default_factory=list)


def train(host: Any) -> None:
    train_config = host.config["train"]
    start_ema(host, train_config)
    if not host.samples:
        raise ValueError("No usable images: every file in the dataset failed to load")
    steps_per_epoch, num_epochs, total_steps = plan_training_steps(host._epoch_size(), train_config)
    host.run_metadata.update(
        steps_per_epoch=steps_per_epoch, planned_total_steps=total_steps, dataset_samples=len(host.samples)
    )

    resume = host.resume_state
    same_data = True
    if resume is not None:
        same_data = resume_position(host, resume)
        if host.global_step >= total_steps:
            logger.warning(
                "The resumed run is already at step %d of %d; raise train.max_steps to continue.",
                host.global_step,
                total_steps,
            )
            return
        # The epochs this run will end at: the ones done so far, then what the remaining steps
        # need at the current epoch size (the planned count assumed every epoch at this size).
        accum = accumulation(train_config)
        num_epochs = host.epoch + math.ceil((total_steps - host.global_step + host.position // accum) / steps_per_epoch)
    host._setup_prior()
    if host.reg_keys:
        host.run_metadata.update({"reg_images": len(host.reg_keys), "reg_lr_multiplier": host.reg_multiplier})

    logger.info(
        "Starting MOSS-Image finetuning: %d epochs, %d total steps; timesteps %s, shift %.2f",
        num_epochs,
        total_steps,
        train_config.get("timestep_distribution", "logit_normal"),
        float(train_config.get("timestep_shift", 1.0)),
    )
    progress_bar = tqdm(
        total=total_steps, initial=host.global_step, desc="Training Steps", disable=not host.is_main_process
    )
    keys = KeyCommands(menu=host._preview_menu, gallery=host._run_gallery if host.gallery.enabled else None)
    monitor = build_monitor(host, train_config)
    if resume is None:
        monitor.exclude(host._maybe_preview(0, total_steps))  # the base model: what later previews compare with
    stopped = _epochs(host, (resume, same_data), (num_epochs, total_steps), (progress_bar, keys, monitor))

    progress_bar.close()
    if keys is not None:
        keys.close()
    if stopped:
        logger.info("MOSS-Image finetuning stopped by key at step %d; resume from the _state folder.", host.global_step)
        return
    host.save_checkpoint("final")
    logger.info("MOSS-Image finetuning completed!")


def _epochs(
    host: Any,
    resumed: tuple[dict[str, Any] | None, bool],
    budget: tuple[int, int],
    tools: tuple[Any, KeyCommands | None, TrainMonitor],
) -> bool:
    """Epochs until the step budget is spent or a key stops the run; True when a key did."""
    resume, same_data = resumed
    num_epochs, total_steps = budget
    acc = Accumulation()
    epoch = host.epoch
    first = resume is not None
    # Until the step budget is spent. num_epochs is only for display; the guard keeps it true
    # should the step count run past the estimate.
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
            loss = _forward_backward(host, order[pos], acc, tools[2])
            if acc.micro % accumulation(host.config["train"]):
                continue
            stopped = _optimizer_step(host, loss, (pos, epoch, num_epochs, total_steps), acc, tools)
            if host.global_step >= total_steps or stopped:
                break
        if stopped:
            return True
        if host.global_step < total_steps:
            host.position = 0
            epoch += 1
    return False


def _forward_backward(host: Any, index: int, acc: Accumulation, monitor: TrainMonitor) -> torch.Tensor:
    """One sample's loss with the prior term when it is due, and its backward (fused Automagic3 steps inside)."""
    key = host.samples[index][0]
    loss: torch.Tensor = host._training_step(index)
    total_loss = loss
    prior_value = None
    if host.prior is not None and host.prior.due(host.global_step):
        x0 = host.latent_cache[key].to(host.device, torch.float32).unsqueeze(0)
        prior_loss = host.prior.loss(host._rel(key), x0, host._predict)
        if prior_loss is not None:
            # One backward for both terms: fused Automagic3 steps inside it.
            total_loss = loss + host.prior.weight * prior_loss
            prior_value = prior_loss.item()
            acc.prior_losses.append(prior_value)
    acc.scales.append(host.reg_multiplier if key in host.reg_keys else 1.0)
    if host.reg_keys and host.optimizer.fused:
        set_step_scale(host.optimizer, acc.scales[-1])  # the update happens inside this backward
    accum = accumulation(host.config["train"])
    cast(Any, total_loss / accum if accum > 1 else total_loss).backward()
    # Regularisation images count in the loss bands too; they are a fifth of the step.
    monitor.record(loss.item(), float(host.last_sigma), prior_value)
    acc.micro += 1
    return loss


def _optimizer_step(
    host: Any,
    loss: torch.Tensor,
    where: tuple[int, int, int, int],
    acc: Accumulation,
    tools: tuple[Any, KeyCommands | None, TrainMonitor],
) -> bool:
    """After the last micro-step of a batch: the optimizer step and EMA, the log line, previews, a due save and the
    keys. True when a key stopped the run."""
    pos, epoch, num_epochs, total_steps = where
    progress_bar, keys, monitor = tools
    if host.reg_keys and not host.optimizer.fused:
        set_step_scale(host.optimizer, sum(acc.scales) / len(acc.scales))
    acc.scales.clear()
    host.step_note = "Prior: %.6f, " % (sum(acc.prior_losses) / len(acc.prior_losses)) if acc.prior_losses else ""
    acc.prior_losses.clear()
    # Fused: every parameter already stepped in backward; this applies the
    # per-group rate votes. Unfused: the whole update happens here.
    host.optimizer.step()
    host.optimizer.zero_grad(set_to_none=True)
    if host.ema is not None:
        host.ema.update()

    host.global_step += 1
    host.position = pos + 1
    progress_bar.update(1)
    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "epoch": f"{epoch + 1}/{num_epochs}"})
    host._log_step(host.global_step, total_steps, loss, epoch, num_epochs)
    monitor.step_done(host.global_step, total_steps, host.optimizer)
    monitor.exclude(host._maybe_preview(host.global_step, total_steps))
    save_every = int(host.config["train"].get("save_every", 0) or 0)
    if save_every > 0 and host.global_step % save_every == 0 and host.global_step < total_steps:
        host.save_checkpoint(host.global_step)
    stopped = keys is not None and keys.handle(host.global_step, host.save_checkpoint)
    if keys is not None:
        monitor.exclude(keys.take_paused_seconds())
    host._watch_config()
    return stopped
