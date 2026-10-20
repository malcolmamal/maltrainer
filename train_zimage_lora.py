#!/usr/bin/env python3
"""
Z-Image LoRA Training Script
Trains LoRA adapters for Z-Image (base) models with YAML configuration support.
"""

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from tqdm import tqdm

from maltrainer.models.lora import LoRANetwork
from maltrainer.models.scheduler import get_flow_match_scheduler
from maltrainer.trainers.zimage import dry_run
from maltrainer.trainers.zimage.data import count_dataset_samples, make_loader
from maltrainer.trainers.zimage.model import load_components
from maltrainer.trainers.zimage.noise import sample_timesteps
from maltrainer.trainers.zimage.optimizer import build_optimizer
from maltrainer.trainers.zimage.step import batch_step
from maltrainer.training.cli import training_arguments
from maltrainer.training.optimizer import linear_warmup
from maltrainer.training.session import log_device, open_training_log
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class ZImageLoRATrainer:
    """Trainer for Z-Image LoRA models (base or turbo)."""

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run, self.dry_run_step_time = dry_run, dry_run_step_time
        self.last_saved_path: str | None = None
        self.lr_scheduler: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        )
        self.is_main_process = self.accelerator.is_main_process
        if self.is_main_process:
            log_device(self.device)
        self.progress_logger = open_training_log(config)

    def load_model(self) -> None:
        """Load Z-Image model components (maltrainer.trainers.zimage.model)."""
        logger.info("Loading Z-Image model...")
        components = load_components(self.config, self.device, self.dtype)
        self.transformer, self.text_encoder, self.tokenizer, self.vae = components
        logger.info("Model loaded successfully")

    def create_lora_network(self) -> None:
        """Create LoRA network for the transformer."""
        logger.info("Creating LoRA network...")
        lora_config = self.config.get("lora", {})
        self.lora_network = LoRANetwork(
            transformer=self.transformer,
            rank=lora_config.get("rank", 4),
            alpha=lora_config.get("alpha", 1.0),
            target_modules=lora_config.get("target_modules", ["to_q", "to_k", "to_v", "to_out"]),
        )
        self.lora_network.prefix = lora_config.get("prefix", "")
        self.lora_network.to(self.device, dtype=self.dtype)
        logger.info(f"LoRA network created with rank={lora_config.get('rank', 4)}")

    def create_dataset(self) -> None:
        """Create training dataset (maltrainer.trainers.zimage.data)."""
        logger.info("Creating dataset...")
        self.train_dataset, self.train_dataloader = make_loader(self.config)
        logger.info(f"Dataset created with {len(self.train_dataset)} samples")

    def _count_dataset_samples(self) -> int:
        return count_dataset_samples(self.config.get("dataset", {}))

    def _sample_timesteps(self, batch_size: int, num_train_timesteps: int) -> torch.Tensor:
        return sample_timesteps(
            self.config.get("train", {}), batch_size, num_train_timesteps, (self.device, self.dtype)
        )

    def dry_run_training(self) -> None:
        """Simulate training without loading or downloading models (maltrainer.trainers.zimage.dry_run)."""
        samples = self._count_dataset_samples()
        self.last_saved_path = dry_run.simulate(
            self.config, samples, self.dry_run_step_time, "lora", self.is_main_process
        )

    def setup_optimizer(self) -> None:
        """Setup optimizer and scheduler."""
        train_config = self.config["train"]
        self.optimizer = build_optimizer(self.lora_network.parameters(), train_config, 1e-4)
        self.noise_scheduler: Any = get_flow_match_scheduler()
        self.lr_scheduler = linear_warmup(self.optimizer, int(train_config.get("learning_rate_warmup_steps", 0) or 0))

    def train(self) -> None:
        """Main training loop."""
        train_config = self.config["train"]
        num_epochs = train_config.get("num_epochs", 10)
        if self.is_main_process:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            print("--- Training Configuration ---")
            print(f"Device: {device_name}")
            print(f"Dtype: {self.dtype}")
            print(f"Transformer: {sum(p.numel() for p in self.transformer.parameters()) / 1e9:.2f}B params")
            print(f"Text Encoder: {sum(p.numel() for p in self.text_encoder.parameters()) / 1e9:.2f}B params")
            print("-------------------------------")

        self.lora_network, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.lora_network, self.optimizer, self.train_dataloader
        )
        steps_per_epoch = math.ceil(len(self.train_dataloader) / max(self.accelerator.gradient_accumulation_steps, 1))
        total_steps = num_epochs * steps_per_epoch
        logger.info("Starting training...")
        progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not self.is_main_process)
        global_step = 0
        for epoch in range(num_epochs):
            self.lora_network.train()
            for batch in self.train_dataloader:
                global_step = batch_step(self, batch, (epoch, num_epochs), (global_step, total_steps), progress_bar)
        progress_bar.close()
        self.save_checkpoint("final")
        logger.info("Training completed!")

    def save_checkpoint(self, step: int | str) -> None:
        """Save LoRA checkpoint."""
        if not self.is_main_process:
            return
        output_dir = Path(self.config["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"zimage_lora_step_{step}.safetensors"
        lora_config = self.config.get("lora", {})
        metadata = {
            "step": str(step),
            "rank": str(lora_config.get("rank", 4)),
            "alpha": str(lora_config.get("alpha", 1.0)),
        }
        save_file(self.lora_network.get_state_dict(), str(save_path), metadata=metadata)
        self.last_saved_path = str(save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    def _log_summary(self, start_time: float) -> None:
        elapsed = max(time.time() - start_time, 0.0)
        output_path = self.last_saved_path or self.config.get("output", {}).get("path", "")
        logger.info("Summary: ran for %.2fs, last output saved to %s", elapsed, output_path)

    def run(self) -> None:
        """Run the complete training pipeline."""
        try:
            set_seed(self.config.get("seed", 42))
            start_time = time.time()
            if self.dry_run:
                self.dry_run_training()
                self._log_summary(start_time)
                return
            self.load_model()
            self.create_lora_network()
            self.create_dataset()
            self.setup_optimizer()
            self.train()
            self._log_summary(start_time)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Exiting...")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            raise e


def main() -> None:
    parser = training_arguments("Train Z-Image LoRA")
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}))
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])
    ZImageLoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
