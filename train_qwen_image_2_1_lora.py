#!/usr/bin/env python3
"""
Qwen-Image-2.1 LoRA Training Script for Maltrainer.
Trains LoRA adapters for Qwen-Image-2.1 using INT8 ConvRot transformer, 3D VAE, and Qwen3-VL text encoder.
Uses automated dataset pre-caching for fast step execution and low VRAM footprint.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from accelerate.utils import set_seed
from tqdm import tqdm

from maltrainer.trainers.qwen_image_2_1.data import make_loader, precache
from maltrainer.trainers.qwen_image_2_1.model import load_lora_model
from maltrainer.trainers.qwen_image_2_1.optimizer import build_optimizer, warmup_scheduler, warmup_steps
from maltrainer.trainers.qwen_image_2_1.step import batch_inputs, flow_loss, plan_training_steps, sample_timesteps
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.training.progress import log_step, simulate_steps
from maltrainer.training.saving import lora_save_path, write_lora
from maltrainer.training.session import log_device, make_accelerator, open_training_log, start_metadata
from maltrainer.utils.config import load_config

logger = logging.getLogger(__name__)


def _say(text: str) -> None:
    print(text, flush=True)


class QwenImage21LoRATrainer:
    """Trainer for Qwen-Image-2.1 LoRA models."""

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run, self.dry_run_step_time = dry_run, dry_run_step_time
        self.last_saved_path: str | None = None
        self.lr_scheduler: Any = None
        self.optimizer: Any = None
        self.run_metadata = start_metadata("qwen_image_2_1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.latent_cache: dict[str, torch.Tensor] = {}
        self.text_cache: dict[str, torch.Tensor] = {}
        self.accelerator = make_accelerator(config)
        self.is_main_process = self.accelerator.is_main_process
        if self.is_main_process:
            log_device(self.device)
        self.progress_logger = open_training_log(config)

    def precache_dataset(self) -> None:
        """Pre-cache VAE latents and Qwen3-VL text embeddings, then unload the VAE and text encoder to free VRAM."""
        caches = precache(self.config, self.device, self.dtype, self.is_main_process, _say)
        self.latent_cache, self.text_cache = caches

    def load_transformer_and_lora(self) -> None:
        """Load INT8 ConvRot / BF16 Qwen-Image-2.1 Transformer Base and inject LoRA network."""
        print("=== Step 2/2: Loading Qwen-Image-2.1 Transformer Base ===", flush=True)
        self.transformer, self.lora_network, self.lora_dtype = load_lora_model(
            self.config, self.device, self.dtype, self.run_metadata
        )
        print("Transformer and LoRA network ready on GPU!", flush=True)

    def create_dataset(self) -> None:
        """Create image training dataset and dataloader."""
        self.train_dataset, self.train_dataloader = make_loader(self.config)

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return sample_timesteps(self.config.get("train", {}), batch_size, self.device)

    def setup_optimizer(self) -> None:
        """Setup optimizer over the LoRA adapters only (maltrainer.trainers.qwen_image_2_1.optimizer)."""
        train_config = self.config["train"]
        self.optimizer, resolved = build_optimizer(list(self.lora_network.trainable_parameters()), train_config)
        if resolved is not None:
            self.run_metadata["optimizer_resolved"] = resolved
        self.lr_scheduler = warmup_scheduler(self.optimizer, warmup_steps(train_config))

    def train(self) -> None:
        """Main Qwen-Image-2.1 LoRA training loop."""
        train_config = self.config["train"]
        self.lora_network, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.lora_network, self.optimizer, self.train_dataloader
        )
        steps_per_epoch, num_epochs, total_steps = plan_training_steps(len(self.train_dataloader), train_config)
        self.run_metadata["steps_per_epoch"] = steps_per_epoch
        self.run_metadata["planned_total_steps"] = total_steps
        self.run_metadata["dataset_samples"] = len(self.train_dataset)
        logger.info("Starting Qwen-Image-2.1 LoRA training for %d epochs (%d total steps)...", num_epochs, total_steps)

        progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not self.is_main_process)
        global_step = 0
        for epoch in range(num_epochs):
            if global_step >= total_steps:
                break
            global_step = self._epoch((epoch, num_epochs), (global_step, total_steps), progress_bar)
        progress_bar.close()
        self.save_checkpoint("final")
        logger.info("Qwen-Image-2.1 LoRA training completed!")

    def _epoch(self, epochs: tuple[int, int], steps: tuple[int, int], progress_bar: Any) -> int:
        """One pass over the loader from step ``steps[0]``, stopping at ``steps[1]``; returns the step reached."""
        (epoch, num_epochs), (global_step, total_steps) = epochs, steps
        save_every = int(self.config["train"].get("save_every", 500) or 0)
        self.lora_network.train()
        for batch in self.train_dataloader:
            with self.accelerator.accumulate(self.lora_network):
                loss = self._batch_loss(batch)
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self._optimizer_step()
            if self.accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                log_step(self.progress_logger, self.optimizer, global_step, total_steps, loss, *epochs, rates=False)
                if save_every > 0 and global_step % save_every == 0 and global_step < total_steps:
                    self.save_checkpoint(global_step)
            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "step": global_step, "epoch": f"{epoch + 1}/{num_epochs}"}
            )
            if global_step >= total_steps:
                break
        return global_step

    def _batch_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        latents, text_embeds = batch_inputs(batch, (self.latent_cache, self.text_cache), self.device, self.dtype)
        loss = flow_loss(self.lora_network, latents, text_embeds, self._sample_timesteps(latents.shape[0]))
        loss_weight_strength = float(self.config["train"].get("loss_weight_strength", 1.0))
        return loss * loss_weight_strength if loss_weight_strength != 1.0 else loss

    def _optimizer_step(self) -> None:
        max_grad_norm = self.config["train"].get("max_grad_norm", 1.0)
        self.accelerator.clip_grad_norm_(self.lora_network.trainable_parameters(), max_grad_norm)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def save_checkpoint(self, step: int | str) -> None:
        """Save LoRA checkpoint safetensors file."""
        if not self.is_main_process:
            return
        output_config = self.config["output"]
        save_path = lora_save_path(output_config, step, "qwen_image_2_1_lora", ("name", "lora_name", "file_name"))
        save_dtype = getattr(torch, str(output_config.get("dtype", "bfloat16")))
        format_type = str(output_config.get("format", "diffusers")).lower()
        network = self.accelerator.unwrap_model(self.lora_network)
        state_dict = network.get_state_dict(save_dtype=save_dtype, format_type=format_type)
        write_lora(self.config, state_dict, save_path, step, self.run_metadata, save_dtype)
        self.last_saved_path = str(save_path)
        logger.info("Saved LoRA checkpoint to %s", save_path)

    def simulate(self) -> None:
        """Dry run: the plan from the real dataset, then a real run's bar and log lines, without models."""
        self.create_dataset()
        train_config = self.config["train"]
        plan = plan_training_steps(len(self.train_dataloader), train_config)
        save_every = int(train_config.get("save_every", 500))
        logger.info(
            "Dry run: %d samples, %d steps/epoch -> %d epochs, %d total steps (%.3fs per simulated step)",
            len(self.train_dataset),
            *plan,
            self.dry_run_step_time,
        )
        simulate_steps(self.progress_logger, plan, self.dry_run_step_time, self.is_main_process, save_every)

    def run(self) -> None:
        """Run training pipeline."""
        try:
            set_seed(self.config.get("seed", 42))
            if self.dry_run:
                self.simulate()
                return
            self.create_dataset()
            self.precache_dataset()
            self.load_transformer_and_lora()
            self.setup_optimizer()
            self.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Exiting...")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error("An error occurred during training: %s", e, exc_info=True)
            raise e


def main() -> None:
    args = training_arguments("Train Qwen-Image-2.1 LoRA in Maltrainer").parse_args()
    config = load_config(args.config)
    start_run(config)
    trainer = QwenImage21LoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time)
    trainer.run()


if __name__ == "__main__":
    main()
