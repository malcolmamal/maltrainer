#!/usr/bin/env python3
"""
Z-Image Textual Inversion Training Script
Trains textual inversion embeddings for Z-Image (base) models with YAML configuration support.
"""

import argparse
import inspect
import json
import logging
import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL
from diffusers.models.transformers import ZImageTransformer2DModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3ForCausalLM

from maltrainer.data.dataset import ZImageDataset
from maltrainer.data.walk import count_files
from maltrainer.models.embedding import TextualInversionEmbedding
from maltrainer.models.scheduler import get_flow_match_scheduler
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class ZImageEmbeddingTrainer:
    """Trainer for Z-Image Textual Inversion embeddings."""

    def __init__(self, config: dict, dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run = dry_run
        self.dry_run_step_time = dry_run_step_time
        self.last_saved_path = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        )

        self.is_main_process = self.accelerator.is_main_process

        # Log device info
        if self.is_main_process:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            logger.info(f"Training on device: {self.device} ({device_name})")
            if self.device.type == "cuda":
                logger.info(
                    f"Total VRAM: {torch.cuda.get_device_properties(self.device).total_memory / 1024**3:.2f} GB"
                )

    def load_model(self):
        """Load Z-Image model components."""
        logger.info("Loading Z-Image model...")

        model_path = self.config["model"]["name_or_path"]

        # Load transformer
        self.transformer = ZImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )
        self.transformer.enable_gradient_checkpointing()

        # Load text encoder
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
        )

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,  # VAE always float32
        )

        # Set requires_grad
        self.transformer.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        # Enable gradients for embedding layer
        self.text_encoder.get_input_embeddings().requires_grad_(True)

        # Move to device
        self.transformer.to(self.device)
        self.text_encoder.to(self.device)
        self.vae.to(self.device)

        logger.info("Model loaded successfully")

    def create_embedding(self):
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

    def create_dataset(self):
        """Create training dataset."""
        logger.info("Creating dataset...")

        dataset_config = self.config["dataset"]
        embedding_config = self.config.get("embedding", {})

        self.train_dataset = ZImageDataset(
            data_root=dataset_config["path"],
            resolution=dataset_config.get("resolution", 512),
            center_crop=dataset_config.get("center_crop", True),
            random_flip=dataset_config.get("random_flip", False),
            trigger_token=embedding_config.get("trigger", "sks"),
            repeats=dataset_config.get("repeats", 1),
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["train"].get("batch_size", 1),
            shuffle=True,
            num_workers=dataset_config.get("num_workers", 0),
        )

        logger.info(f"Dataset created with {len(self.train_dataset)} samples")

    def _count_dataset_samples(self) -> int:
        dataset_config = self.config.get("dataset", {})
        data_root = Path(str(dataset_config.get("path", "")))
        if not data_root.exists():
            logger.warning("Dataset path does not exist: %s", data_root)
            return 1
        supported_exts = {".jpg", ".jpeg", ".png", ".webp"}
        sample_count = count_files(data_root, supported_exts)
        repeats = int(dataset_config.get("repeats", 1) or 1)
        sample_count = max(sample_count, 1) * max(repeats, 1)
        return sample_count

    def dry_run_training(self):
        """Simulate training without loading or downloading models."""
        train_config = self.config.get("train", {})
        num_epochs = int(train_config.get("num_epochs", 10) or 1)
        batch_size = int(train_config.get("batch_size", 1) or 1)
        grad_accum = int(
            train_config.get("gradient_accumulation_steps", self.config.get("gradient_accumulation_steps", 1)) or 1
        )
        num_samples = self._count_dataset_samples()
        steps_per_epoch = math.ceil((num_samples / max(batch_size, 1)) / max(grad_accum, 1))
        total_steps = max(num_epochs * steps_per_epoch, 1)
        save_every = int(train_config.get("save_every", 500) or 0)
        step_time = max(float(self.dry_run_step_time), 0.0)
        output_dir = Path(self.config.get("output", {}).get("path", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Dry run enabled: skipping model download/loading and training compute.")
        logger.info(
            "Simulating %d steps (epochs=%d, steps/epoch=%d, batch_size=%d, grad_accum=%d, step_time=%.2fs).",
            total_steps,
            num_epochs,
            steps_per_epoch,
            batch_size,
            grad_accum,
            step_time,
        )

        start_time = time.time()
        progress_bar = tqdm(
            total=total_steps,
            desc="Dry Run Steps",
            disable=not self.is_main_process,
        )
        for step in range(1, total_steps + 1):
            if step_time > 0:
                time.sleep(step_time)
            progress_bar.update(1)

            if save_every > 0 and step % save_every == 0:
                self._write_dry_run_checkpoint(
                    output_dir=output_dir,
                    step=step,
                    total_steps=total_steps,
                    steps_per_epoch=steps_per_epoch,
                    num_epochs=num_epochs,
                    start_time=start_time,
                )
        progress_bar.close()
        self._write_dry_run_checkpoint(
            output_dir=output_dir,
            step="final",
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
            num_epochs=num_epochs,
            start_time=start_time,
        )
        logger.info("Dry run completed.")

    def _write_dry_run_checkpoint(
        self, output_dir: Path, step, total_steps: int, steps_per_epoch: int, num_epochs: int, start_time: float
    ):
        dataset_config = self.config.get("dataset", {})
        train_config = self.config.get("train", {})
        embedding_config = self.config.get("embedding", {})
        elapsed = max(time.time() - start_time, 0.0)
        completed_steps = total_steps if step == "final" else int(step)
        progress = min(completed_steps / max(total_steps, 1), 1.0)
        eta_seconds = max((total_steps - completed_steps) * float(self.dry_run_step_time), 0.0)

        payload = {
            "mode": "dry_run",
            "kind": "embedding",
            "step": step,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "num_epochs": num_epochs,
            "progress": {
                "completed_steps": completed_steps,
                "percent": round(progress * 100, 2),
                "elapsed_seconds": round(elapsed, 4),
                "eta_seconds": round(eta_seconds, 4),
                "step_time_seconds": float(self.dry_run_step_time),
            },
            "config": self.config,
            "derived": {
                "dataset_path": str(dataset_config.get("path", "")),
                "dataset_resolution": dataset_config.get("resolution", 512),
                "batch_size": train_config.get("batch_size", 1),
                "gradient_accumulation_steps": train_config.get(
                    "gradient_accumulation_steps", self.config.get("gradient_accumulation_steps", 1)
                ),
                "save_every": train_config.get("save_every", 500),
                "dtype": str(self.config.get("dtype", "bfloat16")),
                "mixed_precision": str(self.config.get("mixed_precision", "bf16")),
                "trigger": embedding_config.get("trigger", "sks"),
                "num_vectors": embedding_config.get("num_vectors", 1),
                "output_path": str(output_dir),
            },
        }

        save_path = output_dir / f"zimage_embedding_step_{step}.safetensors"
        save_path.write_text(json.dumps(payload, indent=2))
        self.last_saved_path = str(save_path)

    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        train_config = self.config["train"]

        def _to_float(value, default: float) -> float:
            if value is None:
                return float(default)
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))

        def _to_int(value, default: int) -> int:
            if value is None:
                return int(default)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            return int(str(value))

        def _to_bool(value, default: bool) -> bool:
            if value is None:
                return bool(default)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            value_str = str(value).strip().lower()
            if value_str in {"true", "1", "yes", "y", "on"}:
                return True
            if value_str in {"false", "0", "no", "n", "off"}:
                return False
            return bool(default)

        lr = _to_float(train_config.get("learning_rate", 5e-4), 5e-4)
        beta1 = _to_float(train_config.get("adam_beta1", 0.9), 0.9)
        beta2 = _to_float(train_config.get("adam_beta2", 0.999), 0.999)
        weight_decay = _to_float(train_config.get("weight_decay", 0.01), 0.01)
        eps = _to_float(train_config.get("adam_epsilon", 1e-8), 1e-8)

        # Optimize the entire embedding layer
        # Note: We rely on restore_original_embeddings to keep non-trained tokens frozen
        # Passing slices (non-leaf tensors) caused warnings and prevented updates
        params_to_optimize = [self.text_encoder.get_input_embeddings().weight]

        optimizer_name = str(train_config.get("optimizer", "adamw")).lower()

        if optimizer_name.startswith("prodigy"):
            try:
                from prodigyopt import Prodigy

                beta3_raw = train_config.get("prodigy_beta3", None)
                beta3 = None if beta3_raw is None else _to_float(beta3_raw, 0.0)
                decouple = _to_bool(train_config.get("prodigy_decouple", True), True)
                use_bias_correction = _to_bool(train_config.get("prodigy_use_bias_correction", False), False)
                safeguard_warmup = _to_bool(train_config.get("prodigy_safeguard_warmup", False), False)
                d0 = _to_float(train_config.get("prodigy_d0", 1e-6), 1e-6)
                d_coef = _to_float(train_config.get("prodigy_d_coef", 1.0), 1.0)
                growth_rate = _to_float(train_config.get("prodigy_growth_rate", float("inf")), float("inf"))
                slice_p = _to_int(train_config.get("prodigy_slice_p", 1), 1)

                self.optimizer = Prodigy(
                    params_to_optimize,
                    lr=lr,
                    betas=(beta1, beta2),
                    beta3=beta3,
                    eps=eps,
                    weight_decay=weight_decay,
                    decouple=decouple,
                    use_bias_correction=use_bias_correction,
                    safeguard_warmup=safeguard_warmup,
                    d0=d0,
                    d_coef=d_coef,
                    growth_rate=growth_rate,
                    slice_p=slice_p,
                )
                logger.info("Using optimizer: %s", optimizer_name)
            except (ImportError, OSError, RuntimeError) as e:
                logger.warning(
                    "Failed to load prodigy optimizer (%s). Falling back to AdamW. Error: %s",
                    optimizer_name,
                    e,
                )
                self.optimizer = AdamW(
                    params_to_optimize,
                    lr=lr,
                    betas=(beta1, beta2),
                    weight_decay=weight_decay,
                    eps=eps,
                )
        else:
            self.optimizer = AdamW(
                params_to_optimize,
                lr=lr,
                betas=(beta1, beta2),
                weight_decay=weight_decay,
                eps=eps,
            )

        self.noise_scheduler = get_flow_match_scheduler()

    def restore_original_embeddings(self):
        """Restore original embeddings except for the learned token."""
        with torch.no_grad():
            token_ids = self.embedding.token_ids
            index_no_updates = torch.ones(len(self.tokenizer), dtype=torch.bool)

            for token_id in token_ids:
                index_no_updates[token_id] = False

            # Restore all embeddings except the learned ones
            self.text_encoder.get_input_embeddings().weight[index_no_updates] = self.original_embeddings[
                index_no_updates
            ]

    def train(self):
        """Main training loop."""
        train_config = self.config["train"]
        num_epochs = train_config.get("num_epochs", 10)
        save_every = train_config.get("save_every", 500)

        # Log device and model info
        if self.is_main_process:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            print("--- Training Configuration ---")
            print(f"Device: {device_name}")
            print(f"Dtype: {self.dtype}")
            print(f"Transformer: {sum(p.numel() for p in self.transformer.parameters()) / 1e9:.2f}B params")
            print(f"Text Encoder: {sum(p.numel() for p in self.text_encoder.parameters()) / 1e9:.2f}B params")
            print("-------------------------------")

        # Prepare for training
        self.embedding, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.embedding, self.optimizer, self.train_dataloader
        )

        global_step = 0

        logger.info("Starting training...")

        # Calculate total steps and setup progress bar
        num_update_steps_per_epoch = len(self.train_dataloader)
        max_train_steps = num_epochs * num_update_steps_per_epoch

        progress_bar = tqdm(
            range(max_train_steps),
            desc="Steps",
            disable=not self.is_main_process,
        )

        for epoch in range(num_epochs):
            self.embedding.train()

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.embedding):
                    # Get pixel values and captions from batch
                    pixel_values = batch["pixel_values"].to(self.device)
                    captions = batch["caption"]

                    # Encode with VAE
                    with torch.no_grad():
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        # Z-Image specific scaling/shifting logic
                        zimage_shift = 0.1159
                        zimage_scale = 0.3611
                        latents = (latents - zimage_shift) * zimage_scale
                        latents = latents.to(dtype=self.dtype)

                    # Tokenize
                    tokens = self.tokenizer(
                        [
                            self.tokenizer.apply_chat_template(
                                [{"role": "user", "content": c}],
                                tokenize=False,
                                add_generation_prompt=True,
                                **(
                                    {"enable_thinking": True}
                                    if "enable_thinking"
                                    in inspect.signature(self.tokenizer.apply_chat_template).parameters
                                    or "kwargs" in inspect.signature(self.tokenizer.apply_chat_template).parameters
                                    else {}
                                ),
                            )
                            if hasattr(self.tokenizer, "apply_chat_template")
                            else c
                            for c in captions
                        ],
                        padding="longest",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).to(self.device)

                    # Encode text
                    # We need gradients to flow through text encoder input embeddings
                    base_text_encoder = getattr(self.text_encoder, "model", self.text_encoder)
                    te_out = base_text_encoder(
                        tokens.input_ids,
                        attention_mask=tokens.attention_mask,
                        output_hidden_states=True,
                    )
                    text_embeds = te_out.hidden_states[-2]

                    # Sample noise
                    noise = torch.randn_like(latents)

                    # Sample timestep
                    bsz = latents.shape[0]
                    timesteps = torch.rand(bsz, device=self.device)

                    # Add noise to latents (flow matching formulation)
                    noisy_latents = (1 - timesteps[:, None, None, None]) * latents + timesteps[
                        :, None, None, None
                    ] * noise
                    noisy_latents = noisy_latents.to(dtype=self.dtype)

                    # Predict velocity
                    transformer_kwargs = {}
                    forward_params = inspect.signature(self.transformer.forward).parameters
                    if "cap_mask" in forward_params:
                        transformer_kwargs["cap_mask"] = tokens.attention_mask.bool()

                    output = self.transformer(
                        noisy_latents.unsqueeze(2),
                        1.0 - timesteps,
                        text_embeds,
                        **transformer_kwargs,
                    )

                    # Handle different return types (BaseOutput, tuple, list)
                    if hasattr(output, "sample"):
                        model_pred = output.sample
                    elif isinstance(output, (list, tuple)):
                        model_pred = output[0]
                    else:
                        model_pred = output

                    # If it's still a list (some models return [tensor]), take the first element
                    if isinstance(model_pred, (list, tuple)):
                        model_pred = model_pred[0]

                    # Squeeze the frame dimension if it exists
                    if len(model_pred.shape) == 5:
                        model_pred = model_pred.squeeze(2)

                    # Fix potential channel/batch swap to avoid broadcasting
                    if (
                        model_pred.ndim == 4
                        and model_pred.shape[0] == latents.shape[1]
                        and model_pred.shape[1] == latents.shape[0]
                        and model_pred.shape[2:] == latents.shape[2:]
                    ):
                        model_pred = model_pred.permute(1, 0, 2, 3).contiguous()

                    # Compute loss
                    # Z-Image / Musubi uses target = latents - noise
                    target = latents - noise
                    loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Backpropagate
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        # Clip gradients on the actual weight tensor we are optimizing
                        self.accelerator.clip_grad_norm_(
                            [self.text_encoder.get_input_embeddings().weight], train_config.get("max_grad_norm", 1.0)
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # Restore original embeddings (regularization)
                    self.restore_original_embeddings()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    # Save checkpoint
                    if global_step % save_every == 0:
                        self.save_checkpoint(global_step)

                # Update progress bar logs
                if global_step % 10 == 0:  # Update logs less frequently
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "epoch": f"{epoch + 1}/{num_epochs}"})

        # Save final checkpoint
        self.save_checkpoint("final")
        logger.info("Training completed!")

    def save_checkpoint(self, step):
        """Save embedding checkpoint."""
        if not self.is_main_process:
            return

        output_dir = Path(self.config["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f"zimage_embedding_step_{step}.safetensors"

        # Get embedding vectors
        state_dict = self.embedding.get_state_dict()

        # Add metadata
        embedding_config = self.config.get("embedding", {})
        metadata = {
            "step": str(step),
            "trigger": embedding_config.get("trigger", "sks"),
            "num_vectors": str(embedding_config.get("num_vectors", 1)),
            "init_text": embedding_config.get("init_text", "a photo"),
        }

        save_file(state_dict, str(save_path), metadata=metadata)
        self.last_saved_path = str(save_path)
        logger.info(f"Saved embedding to {save_path}")

    def _log_summary(self, start_time: float):
        elapsed = max(time.time() - start_time, 0.0)
        output_path = self.last_saved_path or self.config.get("output", {}).get("path", "")
        logger.info("Summary: ran for %.2fs, last output saved to %s", elapsed, output_path)

    def run(self):
        """Run the complete training pipeline."""
        try:
            set_seed(self.config.get("seed", 42))
            start_time = time.time()

            if self.dry_run:
                self.dry_run_training()
                self._log_summary(start_time)
                return

            self.load_model()
            self.create_embedding()
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


def main():
    parser = argparse.ArgumentParser(description="Train Z-Image Textual Inversion Embedding")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without downloading/loading models")
    parser.add_argument(
        "--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step during dry run"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    setup_logging(config.get("logging", {}))

    # Save config to output directory
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])

    # Create trainer and run
    trainer = ZImageEmbeddingTrainer(
        config,
        dry_run=args.dry_run,
        dry_run_step_time=args.dry_run_step_time,
    )
    trainer.run()


if __name__ == "__main__":
    main()
