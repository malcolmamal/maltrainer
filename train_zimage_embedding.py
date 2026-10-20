#!/usr/bin/env python3
"""
Z-Image Textual Inversion Training Script
Trains textual inversion embeddings for Z-Image (base) models with YAML configuration support.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import Qwen3ForCausalLM

from maltrainer.models.embedding import TextualInversionEmbedding
from maltrainer.models.scheduler import get_flow_match_scheduler
from maltrainer.trainers.zimage import dry_run
from maltrainer.trainers.zimage.data import count_dataset_samples, image_loader
from maltrainer.trainers.zimage.embedding import batch_loss
from maltrainer.trainers.zimage.model import load_components
from maltrainer.trainers.zimage.optimizer import build_optimizer
from maltrainer.trainers.zimage.pipeline import Pipeline
from maltrainer.training.cli import training_arguments
from maltrainer.training.session import log_device
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class ZImageEmbeddingTrainer(Pipeline):
    """Trainer for Z-Image Textual Inversion embeddings."""

    def _stages(self) -> list[Callable[[], None]]:
        return [self.load_model, self.create_embedding, self.create_dataset, self.setup_optimizer, self.train]

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run, self.dry_run_step_time = dry_run, dry_run_step_time
        self.last_saved_path: str | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        )
        self.is_main_process = self.accelerator.is_main_process
        if self.is_main_process:
            log_device(self.device)

    def load_model(self) -> None:
        """Load Z-Image model components (maltrainer.trainers.zimage.model); the embedding layer trains."""
        logger.info("Loading Z-Image model...")
        components = load_components(self.config, self.device, self.dtype, Qwen3ForCausalLM)
        self.transformer, self.text_encoder, self.tokenizer, self.vae = components
        self.text_encoder.get_input_embeddings().requires_grad_(True)
        logger.info("Model loaded successfully")

    def create_embedding(self) -> None:
        """Create textual inversion embedding."""
        logger.info("Creating textual inversion embedding...")
        embedding_config = self.config.get("embedding", {})
        self.embedding = TextualInversionEmbedding(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            trigger_token=embedding_config.get("trigger", "sks"),
            num_vectors=embedding_config.get("num_vectors", 1),
            init_text=embedding_config.get("init_text", "a photo"),
        )
        # Backup original embeddings for regularization
        self.original_embeddings = self.text_encoder.get_input_embeddings().weight.data.clone()
        logger.info(f"Embedding created with trigger '{embedding_config.get('trigger', 'sks')}'")

    def create_dataset(self) -> None:
        """Create training dataset: the embedding's trigger in front of every caption."""
        logger.info("Creating dataset...")
        trigger = self.config.get("embedding", {}).get("trigger", "sks")
        self.train_dataset, self.train_dataloader = image_loader(self.config, trigger)
        logger.info(f"Dataset created with {len(self.train_dataset)} samples")

    def _count_dataset_samples(self) -> int:
        return count_dataset_samples(self.config.get("dataset", {}))

    def dry_run_training(self) -> None:
        """Simulate training without loading or downloading models (maltrainer.trainers.zimage.dry_run)."""
        embedding_config = self.config.get("embedding", {})
        derived = {
            "trigger": embedding_config.get("trigger", "sks"),
            "num_vectors": embedding_config.get("num_vectors", 1),
        }
        samples = self._count_dataset_samples()
        self.last_saved_path = dry_run.simulate(
            self.config, samples, self.dry_run_step_time, "embedding", self.is_main_process, derived
        )

    def setup_optimizer(self) -> None:
        """Setup optimizer and scheduler.

        The whole embedding layer is optimized; restore_original_embeddings keeps the other tokens frozen
        (passing slices, which are non-leaf tensors, warned and prevented updates).
        """
        params = [self.text_encoder.get_input_embeddings().weight]
        self.optimizer = build_optimizer(params, self.config["train"], 5e-4, lora=False)
        self.noise_scheduler = get_flow_match_scheduler()

    def restore_original_embeddings(self) -> None:
        """Restore original embeddings except for the learned token."""
        with torch.no_grad():
            index_no_updates = torch.ones(len(self.tokenizer), dtype=torch.bool)
            for token_id in self.embedding.token_ids:
                index_no_updates[token_id] = False
            # Restore all embeddings except the learned ones
            self.text_encoder.get_input_embeddings().weight[index_no_updates] = self.original_embeddings[
                index_no_updates
            ]

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

        self.embedding, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.embedding, self.optimizer, self.train_dataloader
        )
        logger.info("Starting training...")
        max_train_steps = num_epochs * len(self.train_dataloader)
        progress_bar = tqdm(range(max_train_steps), desc="Steps", disable=not self.is_main_process)
        global_step = 0
        for epoch in range(num_epochs):
            self.embedding.train()
            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.embedding):
                    loss = batch_loss(self, batch)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        # Clip gradients on the actual weight tensor we are optimizing
                        self.accelerator.clip_grad_norm_(
                            [self.text_encoder.get_input_embeddings().weight], train_config.get("max_grad_norm", 1.0)
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.restore_original_embeddings()  # regularization
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    save_every = int(train_config.get("save_every", 500) or 0)
                    if save_every > 0 and global_step % save_every == 0:
                        self.save_checkpoint(global_step)
                if global_step % 10 == 0:  # Update logs less frequently
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "epoch": f"{epoch + 1}/{num_epochs}"})
        self.save_checkpoint("final")
        logger.info("Training completed!")

    def save_checkpoint(self, step: int | str) -> None:
        """Save embedding checkpoint."""
        if not self.is_main_process:
            return
        output_dir = Path(self.config["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"zimage_embedding_step_{step}.safetensors"
        embedding_config = self.config.get("embedding", {})
        metadata = {
            "step": str(step),
            "trigger": embedding_config.get("trigger", "sks"),
            "num_vectors": str(embedding_config.get("num_vectors", 1)),
            "init_text": embedding_config.get("init_text", "a photo"),
        }
        save_file(self.embedding.get_state_dict(), str(save_path), metadata=metadata)
        self.last_saved_path = str(save_path)
        logger.info(f"Saved embedding to {save_path}")


def main() -> None:
    args = training_arguments("Train Z-Image Textual Inversion Embedding").parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}))
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])
    ZImageEmbeddingTrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
