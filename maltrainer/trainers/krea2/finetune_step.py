"""One step of the Krea 2 finetune: the window for the step, the loss with the prior term when it is due, the step
size of a regularisation image, one fused backward (every trained weight steps inside it), then the log line, the
monitor, previews, a due save, the keys and a look at the config file's sample: section."""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import torch

from maltrainer.finetune.regularisation import set_step_scale
from maltrainer.trainers.krea2.finetune_host import FinetuneHost
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.train_monitor import TrainMonitor

logger = logging.getLogger(__name__)


def _activate(host: FinetuneHost, monitor: TrainMonitor) -> int:
    """The window for this step, swapped in when it changes (the swap is left out of the monitor's speed)."""
    window: int = host._window_at(host.global_step)
    if window != host.rotator.current:
        started = time.time()
        host.rotator.activate(window)
        host.optimizer = host.rotator.optimizer  # what the control API's stats and the progress line read
        monitor.exclude(time.time() - started)
        logger.info(
            "Step %d: training window %s (%.2fB, %.1fs to swap)",
            host.global_step,
            host.rotator.names[window],
            host.rotator.window_parameters(window) / 1e9,
            time.time() - started,
        )
    return window


def _prior_term(host: FinetuneHost, key: str) -> torch.Tensor | None:
    """This sample's prior term, when it is due and the sample has recorded inputs (maltrainer/finetune/prior.py)."""
    if host.prior is None or not host.prior.due(host.global_step):
        return None
    x0 = host.latent_cache[key].to(host.device, torch.float32)  # cached as [1, C, H, W]
    return host.prior.loss(host._rel(key), x0, host._predict)


def _scale_step(host: FinetuneHost, key: str) -> None:
    """Before the fused backward: a regularisation image steps at train.reg_lr_multiplier, a subject at 1."""
    if not host.reg_keys:
        return
    scale = host.reg_multiplier if key in host.reg_keys else 1.0
    set_step_scale(host.rotator.optimizer, scale)
    if host.always_optimizer is not None:
        set_step_scale(host.always_optimizer, scale)


def sample_step(
    host: FinetuneHost,
    index: int,
    where: tuple[int, int, int, int],
    tools: tuple[Any, KeyCommands | None, TrainMonitor],
    shift_logged: list[bool],
) -> bool:
    """One sample, one optimizer step; True when a key stopped the run."""
    pos, epoch, num_epochs, total_steps = where
    progress_bar, keys, monitor = tools
    window = _activate(host, monitor)
    key = host.samples[index][0]
    loss, _, sigma = host._training_step(index, shift_logged)
    prior = _prior_term(host, key)
    total = loss if prior is None or host.prior is None else loss + host.prior.weight * prior
    _scale_step(host, key)
    cast(Any, total).backward()  # fused: every trained weight steps here, the prior term's gradient included
    host.rotator.optimizer.step()  # applies the window's rate votes
    if host.always_optimizer is not None:
        host.always_optimizer.step()
    prior_value = None if prior is None else prior.item()
    host.step_note = "" if prior_value is None else f"Prior: {prior_value:.6f}, "
    monitor.record(loss.item(), float(sigma), prior_value)
    host.global_step += 1
    host.position = pos + 1
    progress_bar.update(1)
    progress_bar.set_postfix(
        {"loss": f"{loss.item():.4f}", "window": host.rotator.names[window], "epoch": f"{epoch + 1}/{num_epochs}"}
    )
    host._log_step(host.global_step, total_steps, loss, epoch, num_epochs)
    monitor.step_done(host.global_step, total_steps, host.rotator.optimizer)
    monitor.exclude(host._maybe_preview(host.global_step, total_steps))
    save_every = int(host.config["train"].get("save_every", 0) or 0)
    if save_every > 0 and host.global_step % save_every == 0 and host.global_step < total_steps:
        host.save_checkpoint(host.global_step)
    stopped = keys is not None and keys.handle(host.global_step, host.save_checkpoint)
    if keys is not None:
        monitor.exclude(keys.take_paused_seconds())
    host._watch_config()
    return stopped
