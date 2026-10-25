"""The image LoRA trainers' step loop: epochs of micro-steps, an optimizer step every accumulation, then the
log line, a due save and the key commands. ``ImageLoRATrainer`` (image_lora.py) supplies what it reads."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
from tqdm import tqdm

from maltrainer.control.launch import make_keys
from maltrainer.control.stats import RunView
from maltrainer.training.progress import log_step
from maltrainer.training.steps import plan_training_steps
from maltrainer.utils.ema import OffloadedParameterEMA, ParameterEMA
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.train_monitor import TrainMonitor

logger = logging.getLogger(__name__)


class StepLoop:
    label: ClassVar[str]  # the model in the log lines: "Starting <label> LoRA training ..."
    key_commands: ClassVar[bool] = True  # s / q / p / h at the step boundaries
    schedule: ClassVar[tuple[str, float] | None] = None  # timestep distribution and shift defaults, for the start line

    config: dict[str, Any]
    accelerator: Any
    lora_network: Any
    optimizer: Any
    lr_scheduler: Any
    ema: ParameterEMA | OffloadedParameterEMA | None
    run_metadata: dict[str, object]
    samples: list[tuple[str, str]]
    is_main_process: bool
    progress_logger: logging.Logger
    # The health monitor (train.monitor_every): a [monitor] log line each window, and the loss, seconds per step and
    # time left of the control API. ``epoch`` is the running one, 0-based, as the control API reads it.
    monitor: TrainMonitor
    epoch: int = 0

    def _epoch_order(self) -> list[int]:
        raise NotImplementedError

    def _training_step(self, index: int) -> Any:  # one sample's loss (Krea 2: and the noised latent and sigma)
        raise NotImplementedError

    def save_checkpoint(self, step: int | str) -> bool | None:
        raise NotImplementedError

    def train(self) -> None:
        train_config = self.config["train"]
        self._save_every = int(train_config.get("save_every", 250) or 0)
        self.lora_network, self.optimizer = self.accelerator.prepare(self.lora_network, self.optimizer)
        network = self.accelerator.unwrap_model(self.lora_network)
        self._start_ema(network, train_config)
        steps_per_epoch, num_epochs, total_steps = plan_training_steps(len(self._epoch_order()), train_config)
        self.run_metadata.update(
            steps_per_epoch=steps_per_epoch, planned_total_steps=total_steps, dataset_samples=len(self.samples)
        )
        self._announce(num_epochs, total_steps)
        self.monitor = TrainMonitor(every=int(train_config.get("monitor_every", 100) or 100))

        progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not self.is_main_process)
        keys = self._make_keys() if self.is_main_process else None
        self._before_loop(total_steps)
        global_step, stopped = self._loop(network, num_epochs, total_steps, progress_bar, keys)
        progress_bar.close()
        if keys is not None:
            keys.close()
        if stopped:
            logger.info(
                "%s LoRA training stopped by key at step %d (saved as step %d).", self.label, global_step, global_step
            )
            return
        self.save_checkpoint("final")
        logger.info("%s LoRA training completed!", self.label)

    def _start_ema(self, network: Any, train_config: dict[str, Any]) -> None:
        ema_decay = float(train_config.get("ema_decay", 0.0) or 0.0)
        if ema_decay > 0:
            warmup = bool(train_config.get("ema_warmup", True))
            self.ema = ParameterEMA(network.trainable_parameters(), decay=ema_decay, warmup=warmup)
            self.run_metadata["ema_decay"] = ema_decay
            logger.info("EMA of the LoRA weights: decay %.4f (checkpoints save the average)", ema_decay)

    def _announce(self, num_epochs: int, total_steps: int) -> None:
        if self.schedule is None:
            logger.info("Starting %s LoRA training: %d epochs, %d total steps", self.label, num_epochs, total_steps)
            return
        train_config = self.config["train"]
        distribution, shift = self.schedule
        self.run_metadata["resolved_sigma_shift"] = float(train_config.get("timestep_shift", shift))
        logger.info(
            "Starting %s LoRA training: %d epochs, %d total steps; timesteps %s, shift %.2f",
            self.label,
            num_epochs,
            total_steps,
            train_config.get("timestep_distribution", distribution),
            self.run_metadata["resolved_sigma_shift"],
        )

    def _make_keys(self) -> KeyCommands | None:
        return make_keys(KeyCommands, self.config, RunView(self)) if self.key_commands else None

    def _loop(
        self, network: Any, num_epochs: int, total_steps: int, bar: Any, keys: KeyCommands | None
    ) -> tuple[int, bool]:
        """The epochs, until total_steps optimizer steps ran or a key stopped the run: (steps run, stopped)."""
        global_step, stopped = 0, False
        for epoch in range(num_epochs):
            global_step, stopped = self._epoch(network, (epoch, num_epochs), (global_step, total_steps), bar, keys)
            if global_step >= total_steps or stopped:
                break
        return global_step, stopped

    def _epoch(
        self, network: Any, epochs: tuple[int, int], steps: tuple[int, int], bar: Any, keys: KeyCommands | None
    ) -> tuple[int, bool]:
        """One epoch's samples from step ``steps[0]`` on, until ``steps[1]`` or a key stops it: (step, stopped)."""
        (epoch, num_epochs), (global_step, total_steps) = epochs, steps
        stopped = False
        self.epoch = epoch
        self.lora_network.train()
        for index in self._epoch_order():
            with self.accelerator.accumulate(self.lora_network):
                loss = self._micro_step(index)
                self._record_loss(loss)
                if self.accelerator.sync_gradients:
                    self._optimizer_step(network)
            if self.accelerator.sync_gradients:
                global_step += 1
                stopped = self._step_done(global_step, total_steps, loss, epochs, bar, keys)
            bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step, "epoch": f"{epoch + 1}/{num_epochs}"})
            if global_step >= total_steps or stopped:
                break
        return global_step, stopped

    def _micro_step(self, index: int) -> torch.Tensor:
        """One sample's forward and backward; returns its loss."""
        loss: torch.Tensor = self._training_step(index)
        self.accelerator.backward(loss)
        return loss

    def _record_loss(self, loss: torch.Tensor) -> None:
        """The micro-step's loss into the monitor, with its sigma when the trainer keeps it as ``last_sigma``."""
        sigma = getattr(self, "last_sigma", None)
        self.monitor.record(loss.item(), None if sigma is None else float(sigma))

    def _optimizer_step(self, network: Any) -> None:
        max_grad_norm = float(self.config["train"].get("max_grad_norm", 1.0))
        self.accelerator.clip_grad_norm_(network.trainable_parameters(), max_grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        if self.ema is not None:
            self.ema.update()

    def _step_done(
        self, step: int, total: int, loss: torch.Tensor, epochs: tuple[int, int], bar: Any, keys: KeyCommands | None
    ) -> bool:
        """After an optimizer step: the bar, the log line, a due save, the keys. True when a key stopped the run."""
        bar.update(1)
        self._log_step(step, total, loss, *epochs)
        self.monitor.step_done(step, total, self.optimizer)
        self._after_log(step, total)
        if self._save_every > 0 and step % self._save_every == 0 and step < total:
            self.save_checkpoint(step)
        stopped = keys is not None and keys.handle(step, self.save_checkpoint)
        if keys is not None:
            self.monitor.exclude(keys.take_paused_seconds())
        self._after_keys(keys)
        return stopped

    def _before_loop(self, total_steps: int) -> None:
        """Before the first step (MOSS-Image: the step-0 previews)."""

    def _after_log(self, step: int, total: int) -> None:
        """After a step's log line and the monitor's, before its save (MOSS-Image: the previews)."""

    def _after_keys(self, keys: KeyCommands | None) -> None:
        """After the keys were handled and the paused time left out of the monitor (MOSS-Image: the config reload)."""

    def _log_step(self, step: int, total: int, loss: torch.Tensor, epoch: int, num_epochs: int) -> None:
        log_step(self.progress_logger, self.optimizer, step, total, loss, epoch, num_epochs, self._step_note())

    def _step_note(self) -> str:
        """Text the log line carries between the epoch and the rate (``step_note``: the finetune's)."""
        return str(getattr(self, "step_note", ""))
