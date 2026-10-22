"""The MiniMax H3 LoRA trainer's step loop: every epoch runs every sample (no early stop at the planned total),
an optimizer step each accumulation, its progress line with the per-stream losses, and the due saves."""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

from tqdm import tqdm

from maltrainer.models.minimax_h3.packing import AUDIO_SIGMA_SHIFT, VIDEO_SIGMA_SHIFT
from maltrainer.trainers.minimax_h3.sample import H3Sample
from maltrainer.trainers.minimax_h3.step import StepHost
from maltrainer.training.progress import log_step
from maltrainer.utils.ema import ParameterEMA

logger = logging.getLogger(__name__)


class LoopHost(StepHost, Protocol):
    """What the loop reads from the trainer; the methods stay the trainer's (tests patch them there)."""

    accelerator: Any
    lora_network: Any
    optimizer: Any
    lr_scheduler: Any
    ema: ParameterEMA | None
    samples: list[H3Sample]
    run_metadata: dict[str, object]
    progress_logger: logging.Logger
    is_main_process: bool

    def _epoch_order(self) -> list[int]: ...
    def _training_step(self, sample: H3Sample, flow_shift: float) -> Any: ...
    def _resolve_flow_shift(self, latent_h: int, latent_w: int) -> float: ...
    def save_checkpoint(self, step: int | str) -> None: ...


def train(host: LoopHost) -> None:
    """Main MiniMax H3 LoRA training loop (~0.8s/step)."""
    train_config = host.config["train"]
    num_epochs = train_config.get("num_epochs", 10)
    host.lora_network, host.optimizer = host.accelerator.prepare(host.lora_network, host.optimizer)
    ema_decay = float(train_config.get("ema_decay", 0.0) or 0.0)
    if ema_decay > 0:
        params = host.accelerator.unwrap_model(host.lora_network).trainable_parameters()
        host.ema = ParameterEMA(params, decay=ema_decay, warmup=bool(train_config.get("ema_warmup", True)))
        host.run_metadata["ema_decay"] = ema_decay
        logger.info("EMA of the LoRA weights: decay %.4f (checkpoints save the average)", ema_decay)

    grad_accum = max(host.accelerator.gradient_accumulation_steps, 1)
    items_per_epoch = len(host._epoch_order())
    total_steps = max(num_epochs * items_per_epoch // grad_accum, 1)  # no sync ends an epoch
    host.run_metadata["steps_per_epoch"] = math.ceil(items_per_epoch / grad_accum)
    host.run_metadata["planned_total_steps"] = total_steps
    host.run_metadata["dataset_samples"] = len(host.samples)
    logger.info(
        "Starting accelerated MiniMax H3 LoRA training for %d epochs (%d total steps)...", num_epochs, total_steps
    )
    shift_logged = [False]
    progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not host.is_main_process)
    global_step = 0
    for epoch in range(num_epochs):
        host.lora_network.train()
        for index in host._epoch_order():
            sample = host.samples[index]
            flow_shift = _flow_shift(host, sample, shift_logged)
            global_step = _sample_step(
                host, (sample, flow_shift), (epoch, num_epochs), (global_step, total_steps), progress_bar
            )
    progress_bar.close()
    host.save_checkpoint("final")
    logger.info("MiniMax H3 LoRA training completed!")


def _sample_step(
    host: LoopHost, item: tuple[H3Sample, float], epochs: tuple[int, int], steps: tuple[int, int], bar: Any
) -> int:
    """One sample's micro-step (and the optimizer step when the accumulation completes); returns the step."""
    train_config = host.config["train"]
    (sample, flow_shift), (global_step, total_steps) = item, steps
    with host.accelerator.accumulate(host.lora_network):
        loss, parts = host._training_step(sample, flow_shift)
        host.accelerator.backward(loss)
        if host.accelerator.sync_gradients:
            host.accelerator.clip_grad_norm_(
                host.lora_network.trainable_parameters(), train_config.get("max_grad_norm", 1.0)
            )
            host.optimizer.step()
            if host.lr_scheduler is not None:
                host.lr_scheduler.step()
            host.optimizer.zero_grad()
            if host.ema is not None:
                host.ema.update()
    if host.accelerator.sync_gradients:
        global_step += 1
        bar.update(1)
        detail = "".join(f", {name}: {parts[name]:.6f}" for name in ("video", "audio") if name in parts)
        # Prodigy's group lr is only the multiplier; the line shows lr * d, so a d stuck near d0 shows early.
        suffix = f"{detail} [{sample.kind}]"
        log_step(
            host.progress_logger, host.optimizer, global_step, total_steps, loss, *epochs, rates=False, suffix=suffix
        )
        save_every = int(train_config.get("save_every", 500) or 0)
        if save_every > 0 and global_step % save_every == 0 and global_step < total_steps:
            host.save_checkpoint(global_step)
    bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step, "epoch": f"{epochs[0] + 1}/{epochs[1]}"})
    return global_step


def _flow_shift(host: LoopHost, sample: H3Sample, logged: list[bool]) -> float:
    """The sample's sigma shift; the run's first one is logged and recorded."""
    _, _, _, h_lat, w_lat = sample.video.shape
    flow_shift = host._resolve_flow_shift(h_lat, w_lat)
    if not logged[0]:
        logged[0] = True
        host.run_metadata["resolved_sigma_shift"] = round(flow_shift, 6)
        host.run_metadata["latent_size"] = f"{h_lat}x{w_lat}"
        logger.info(
            "Timestep schedule: %s, sigma shift %.3f (latent %dx%d); audio sigma follows "
            "the inference coupling (video shift %.0f -> audio shift %.0f)",
            host.config["train"].get("timestep_distribution", "krea2_shift"),
            flow_shift,
            h_lat,
            w_lat,
            VIDEO_SIGMA_SHIFT,
            AUDIO_SIGMA_SHIFT,
        )
    return flow_shift
