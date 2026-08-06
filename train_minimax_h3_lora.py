#!/usr/bin/env python3
"""
MiniMax H3 LoRA Training Script for Maltrainer.
Trains LoRA adapters for MiniMax H3 using INT8 ConvRot transformer, FP16 Video VAE, and Qwen3-VL text encoder.
Uses automated dataset pre-caching for 40x step acceleration (~0.8s/step).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator, init_empty_weights
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from maltrainer.models.minimax_h3.int8_convrot import (
    load_comfy_int8_convrot_state_dict,
    prepare_int8_convrot_modules,
    enable_int8_convrot,
)
from maltrainer.models.minimax_h3.model import (
    MiniMaxH3Transformer,
    MiniMaxH3TransformerConfig,
    MiniMaxH3TokenTag,
)
from maltrainer.models.minimax_h3.video_vae import load_minimax_h3_video_vae
from maltrainer.models.minimax_h3.text_encoder import load_minimax_h3_text_encoder
from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRANetwork
from maltrainer.data.minimax_dataset import MiniMaxH3ImageDataset
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def patchify_video(latent: torch.Tensor, patch_size=(1, 2, 2)) -> torch.Tensor:
    """Patchify video/image latents matching ComfyUI / MiniMax H3 layout [B, C, T, H, W] -> [B, N_tokens, C*1*2*2]."""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    x = torch.einsum("nctrhpwq->nthwcrpq", x)
    return x.reshape(b, t * h * w, c * pt * ph * pw)


def apply_flow_shift(t: torch.Tensor, shift: float = 12.0) -> torch.Tensor:
    """Apply discrete flow shift mapping: t_shifted = (shift * t) / (1 + (shift - 1) * t)."""
    if shift == 1.0:
        return t
    return (shift * t) / (1.0 + (shift - 1.0) * t)


class MiniMaxH3LoRATrainer:
    """Trainer for MiniMax H3 LoRA models."""

    def __init__(self, config: dict, dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run = dry_run
        self.dry_run_step_time = dry_run_step_time
        self.last_saved_path = None
        self.lr_scheduler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))

        self.latent_cache: Dict[str, torch.Tensor] = {}
        self.text_cache: Dict[str, torch.Tensor] = {}

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("train", {}).get("gradient_accumulation_steps", 1),
        )

        self.is_main_process = self.accelerator.is_main_process

        if self.is_main_process:
            device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            logger.info("Training on device: %s (%s)", self.device, device_name)
            if self.device.type == "cuda":
                logger.info("Total VRAM: %.2f GB", torch.cuda.get_device_properties(self.device).total_memory / 1024**3)

        log_dir = Path(config.get("output", {}).get("path", "./output"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)

        self.progress_logger = logging.getLogger("progress")
        self.progress_logger.setLevel(logging.INFO)
        self.progress_logger.addHandler(file_handler)
        self.progress_logger.propagate = False

    def precache_dataset(self):
        """Pre-cache VAE latents and Qwen3-VL text embeddings into memory, then unload VAE & Text Encoder to free VRAM."""
        model_config = self.config.get("model", {})
        vae_path = model_config["vae_path"]
        text_encoder_path = model_config["text_encoder_path"]
        dataset_config = self.config["dataset"]

        print("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text Embeddings) ===", flush=True)
        logger.info("Starting dataset pre-caching...")

        # Create temporary dataset without repeats for unique extraction
        temp_dataset = MiniMaxH3ImageDataset(
            data_root=dataset_config["path"],
            resolution=dataset_config.get("resolution", 512),
            trigger_token=dataset_config.get("trigger", None),
            default_caption=dataset_config.get("default_caption", ""),
            repeats=1,
        )

        # 1. Load VAE & Encode All Unique Images
        print(f"Loading Video VAE from {vae_path} for pre-caching...", flush=True)
        vae = load_minimax_h3_video_vae(vae_path, device=self.device)

        temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False)
        print("Caching VAE latents for images...", flush=True)
        for item in tqdm(temp_loader, desc="Caching VAE Latents", disable=not self.is_main_process):
            img_path = item["image_path"][0]
            if img_path not in self.latent_cache:
                pixel_values = item["pixel_values"].to(self.device)
                with torch.no_grad():
                    latents = vae.encode_image(pixel_values).detach().cpu()
                self.latent_cache[img_path] = latents

        del vae
        gc.collect()
        torch.cuda.empty_cache()
        print("Video VAE latents pre-cached and VAE unloaded from VRAM.", flush=True)

        # 2. Load Text Encoder & Encode All Unique Captions
        print(f"Loading Text Encoder from {text_encoder_path} for pre-caching...", flush=True)
        text_encoder = load_minimax_h3_text_encoder(text_encoder_path, device=self.device, dtype=self.dtype)

        unique_captions = list(set(temp_dataset.captions))
        print(f"Caching text embeddings for {len(unique_captions)} unique prompts...", flush=True)
        for caption in tqdm(unique_captions, desc="Caching Prompts", disable=not self.is_main_process):
            if caption not in self.text_cache:
                with torch.no_grad():
                    text_embeds = text_encoder.encode_prompts([caption], device=self.device, dtype=self.dtype).detach().cpu()
                self.text_cache[caption] = text_embeds

        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        print("Text embeddings pre-cached and Text Encoder unloaded from VRAM.", flush=True)
        logger.info("Pre-cached %d latents and %d prompt embeddings. Unloaded feature extractors from VRAM.", len(self.latent_cache), len(self.text_cache))

    def load_transformer_and_lora(self):
        """Load INT8 ConvRot transformer base and inject LoRA network."""
        print("=== Step 2/2: Loading MiniMax H3 INT8 Transformer Base ===", flush=True)
        logger.info("Loading Transformer base and LoRA network...")
        model_config = self.config.get("model", {})
        transformer_path = model_config["transformer_path"]

        int8_state_dict, num_quant_layers = load_comfy_int8_convrot_state_dict(
            transformer_path, device=torch.device("cpu")
        )
        logger.info("Parsed %d INT8 ConvRot quantized layers", num_quant_layers)

        if "adaln_t_table" in int8_state_dict:
            table = int8_state_dict["adaln_t_table"]
            table_size, embed_dim = table.shape[0], table.shape[1]
            logger.info("Detected pruned AdaLN table: size=%d, embed_dim=%d", table_size, embed_dim)
            transformer_cfg = MiniMaxH3TransformerConfig(adaln_t_table_size=table_size, time_embed_dim=embed_dim)
        else:
            transformer_cfg = MiniMaxH3TransformerConfig()

        print("Instantiating Transformer structure on GPU...", flush=True)
        with init_empty_weights(include_buffers=True):
            transformer = MiniMaxH3Transformer(transformer_cfg)

        prepare_int8_convrot_modules(transformer, int8_state_dict)
        transformer.load_state_dict(int8_state_dict, strict=False, assign=True)
        enable_int8_convrot(transformer)

        transformer.enable_gradient_checkpointing()
        transformer.requires_grad_(False)
        transformer.to(self.device)
        self.transformer = transformer

        # Inject LoRA
        lora_config = self.config.get("lora", {})
        self.lora_network = MiniMaxH3LoRANetwork(
            transformer=self.transformer,
            rank=lora_config.get("rank", 16),
            alpha=lora_config.get("alpha", 16.0),
            prefix=lora_config.get("prefix", "diffusion_model."),
            target_modules=lora_config.get("target_modules", ["qkv_proj", "out_proj", "fc1", "fc2"]),
        )
        self.lora_network.to(self.device, dtype=self.dtype)
        print("Transformer and LoRA network ready on GPU!", flush=True)

    def create_dataset(self):
        """Create image training dataset and dataloader."""
        dataset_config = self.config["dataset"]

        self.train_dataset = MiniMaxH3ImageDataset(
            data_root=dataset_config["path"],
            resolution=dataset_config.get("resolution", 512),
            trigger_token=dataset_config.get("trigger", None),
            default_caption=dataset_config.get("default_caption", ""),
            repeats=dataset_config.get("repeats", 1),
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["train"].get("batch_size", 1),
            shuffle=True,
            num_workers=dataset_config.get("num_workers", 0),
        )

        logger.info("Training dataset configured with %d samples", len(self.train_dataset))

    def _count_dataset_samples(self) -> int:
        dataset_config = self.config.get("dataset", {})
        data_root = Path(str(dataset_config.get("path", "")))
        if not data_root.exists():
            return 1
        supported_exts = {".jpg", ".jpeg", ".png", ".webp"}
        sample_count = 0
        for root, _, files in os.walk(data_root):
            for fname in files:
                if Path(fname).suffix.lower() in supported_exts:
                    sample_count += 1
        repeats = int(dataset_config.get("repeats", 1) or 1)
        return max(sample_count, 1) * max(repeats, 1)

    def dry_run_training(self):
        """Simulate training without downloading or loading models."""
        train_config = self.config.get("train", {})
        num_epochs = int(train_config.get("num_epochs", 10) or 1)
        batch_size = int(train_config.get("batch_size", 1) or 1)
        grad_accum = int(train_config.get("gradient_accumulation_steps", 1) or 1)
        num_samples = self._count_dataset_samples()
        steps_per_epoch = math.ceil((num_samples / max(batch_size, 1)) / max(grad_accum, 1))
        total_steps = max(num_epochs * steps_per_epoch, 1)
        save_every = int(train_config.get("save_every", 500) or 0)
        step_time = max(float(self.dry_run_step_time), 0.0)
        output_dir = Path(self.config.get("output", {}).get("path", "./output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Dry run enabled: simulating %d steps.", total_steps)

        start_time = time.time()
        progress_bar = tqdm(total=total_steps, desc="Dry Run Steps", disable=not self.is_main_process)
        for step in range(1, total_steps + 1):
            if step_time > 0:
                time.sleep(step_time)
            progress_bar.update(1)

            if save_every > 0 and step % save_every == 0:
                self._write_dry_run_checkpoint(output_dir, step, total_steps, steps_per_epoch, num_epochs, start_time)

        progress_bar.close()
        self._write_dry_run_checkpoint(output_dir, "final", total_steps, steps_per_epoch, num_epochs, start_time)
        logger.info("Dry run completed successfully.")

    def _write_dry_run_checkpoint(self, output_dir: Path, step, total_steps: int, steps_per_epoch: int, num_epochs: int, start_time: float):
        elapsed = max(time.time() - start_time, 0.0)
        completed_steps = total_steps if step == "final" else int(step)
        progress = min(completed_steps / max(total_steps, 1), 1.0)

        payload = {
            "mode": "dry_run",
            "kind": "minimax_h3_lora",
            "step": step,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "num_epochs": num_epochs,
            "progress": {
                "completed_steps": completed_steps,
                "percent": round(progress * 100, 2),
                "elapsed_seconds": round(elapsed, 4),
            },
            "config": self.config,
        }

        save_path = output_dir / f"minimax_h3_lora_step_{step}.safetensors"
        save_path.write_text(json.dumps(payload, indent=2))
        self.last_saved_path = str(save_path)

    def setup_optimizer(self):
        """Setup optimizer."""
        train_config = self.config["train"]
        lr = float(train_config.get("learning_rate", 1e-4))
        optimizer_name = str(train_config.get("optimizer", "adamw8bit")).lower()

        if optimizer_name in {"adamw8bit", "adam8bit"}:
            try:
                import bitsandbytes as bnb
                optimizer_cls = bnb.optim.AdamW8bit if optimizer_name == "adamw8bit" else bnb.optim.Adam8bit
                self.optimizer = optimizer_cls(self.lora_network.parameters(), lr=lr)
                logger.info("Using 8-bit optimizer: %s", optimizer_name)
            except Exception as e:
                logger.warning("Bitsandbytes not available (%s). Falling back to AdamW.", e)
                self.optimizer = AdamW(self.lora_network.parameters(), lr=lr)
        elif optimizer_name.startswith("prodigy"):
            try:
                from prodigyopt import Prodigy
                prodigy_lr = float(train_config.get("learning_rate", 1.0))
                if prodigy_lr < 0.01:
                    prodigy_lr = 1.0
                weight_decay = float(train_config.get("weight_decay", 0.01))
                self.optimizer = Prodigy(
                    self.lora_network.parameters(),
                    lr=prodigy_lr,
                    weight_decay=weight_decay,
                    d0=1e-6,
                    d_coef=1.0,
                    use_bias_correction=True,
                    safeguard_warmup=True,
                )
                logger.info("Using Prodigy adaptive optimizer (lr=%.1f, weight_decay=%.3f)", prodigy_lr, weight_decay)
            except Exception as e:
                logger.warning("Prodigy not available (%s). Falling back to AdamW.", e)
                self.optimizer = AdamW(self.lora_network.parameters(), lr=lr)
        else:
            self.optimizer = AdamW(self.lora_network.parameters(), lr=lr)

    def train(self):
        """Main MiniMax H3 LoRA training loop (~0.8s/step)."""
        train_config = self.config["train"]
        num_epochs = train_config.get("num_epochs", 10)
        save_every = train_config.get("save_every", 500)
        flow_shift = float(train_config.get("timestep_shift", 12.0))

        self.lora_network, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.lora_network, self.optimizer, self.train_dataloader
        )

        global_step = 0
        grad_accum = self.accelerator.gradient_accumulation_steps
        steps_per_epoch = math.ceil(len(self.train_dataloader) / max(grad_accum, 1))
        total_steps = num_epochs * steps_per_epoch

        logger.info("Starting accelerated MiniMax H3 LoRA training for %d epochs (%d total steps)...", num_epochs, total_steps)

        progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not self.is_main_process)

        for epoch in range(num_epochs):
            self.lora_network.train()

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.lora_network):
                    image_paths = batch["image_path"]
                    captions = batch["caption"]

                    # 1. Fetch pre-cached latents [B, 24, 1, H//16, W//16]
                    latents = torch.cat([self.latent_cache[p] for p in image_paths], dim=0).to(device=self.device, dtype=self.dtype)

                    # 2. Fetch pre-cached text embeddings [B, S, 5120]
                    text_embeds = torch.cat([self.text_cache[c] for c in captions], dim=0).to(device=self.device, dtype=self.dtype)

                    # 3. Sample timesteps & apply discrete flow shift (sigma)
                    bsz = latents.shape[0]
                    raw_timesteps = torch.rand(bsz, device=self.device, dtype=self.dtype)
                    shifted_sigmas = apply_flow_shift(raw_timesteps, shift=flow_shift)

                    # 4. Sample noise and create noisy latents: x_t = (1 - sigma)*x0 + sigma*noise
                    noise = torch.randn_like(latents)
                    sigma_broadcast = shifted_sigmas.view(bsz, 1, 1, 1, 1)
                    noisy_latents = (1.0 - sigma_broadcast) * latents + sigma_broadcast * noise

                    # In MiniMax H3, the transformer input timestep coordinate is 1.0 - sigma
                    model_timesteps = 1.0 - shifted_sigmas

                    # 5. Build sequence inputs for MiniMax H3 transformer
                    _, c_lat, f_lat, h_lat, w_lat = latents.shape
                    num_video_tokens = f_lat * (h_lat // 2) * (w_lat // 2)
                    num_text_tokens = text_embeds.shape[1]
                    seq_len = num_text_tokens + num_video_tokens

                    video_patches = patchify_video(noisy_latents)
                    dummy_audio = torch.zeros((bsz, 0, 32), device=self.device, dtype=self.dtype)

                    text_indices = torch.arange(0, num_text_tokens, device=self.device)
                    video_indices = torch.arange(num_text_tokens, seq_len, device=self.device)
                    audio_indices = torch.zeros((0,), device=self.device, dtype=torch.long)

                    timestep_indices = torch.zeros((seq_len,), device=self.device, dtype=torch.long)
                    token_tags = torch.zeros((seq_len,), device=self.device, dtype=torch.long)
                    token_tags[text_indices] = MiniMaxH3TokenTag.TEXT
                    token_tags[video_indices] = MiniMaxH3TokenTag.VIDEO

                    grid_h, grid_w = h_lat // 2, w_lat // 2
                    pos_h = torch.arange(grid_h, device=self.device).repeat_interleave(grid_w)
                    pos_w = torch.arange(grid_w, device=self.device).repeat(grid_h)
                    pos_t = torch.zeros(num_video_tokens, device=self.device, dtype=torch.long)
                    video_pos = torch.stack([pos_t, pos_h, pos_w], dim=1)

                    text_pos = torch.zeros((num_text_tokens, 3), device=self.device, dtype=torch.long)
                    position_ids = torch.cat([text_pos, video_pos], dim=0)

                    # 6. Model forward pass
                    model_output = self.lora_network(
                        video_hidden_states=video_patches,
                        audio_hidden_states=dummy_audio,
                        encoder_hidden_states=text_embeds,
                        timestep=model_timesteps,
                        timestep_indices=timestep_indices,
                        token_tags=token_tags,
                        position_ids=position_ids,
                        video_indices=video_indices,
                        audio_indices=audio_indices,
                        text_indices=text_indices,
                    )

                    v_pred_patches = model_output.video

                    target_velocity = latents - noise
                    target_patches = patchify_video(target_velocity)

                    loss = torch.nn.functional.mse_loss(v_pred_patches.float(), target_patches.float())

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.lora_network.parameters(), train_config.get("max_grad_norm", 1.0))
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.progress_logger.info("Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, LR: %.2e",
                                                global_step, total_steps, int(global_step/total_steps*100), loss.item(), epoch + 1, num_epochs, current_lr)

                    if global_step % save_every == 0:
                        self.save_checkpoint(global_step)

                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step, "epoch": f"{epoch + 1}/{num_epochs}"})

        progress_bar.close()
        self.save_checkpoint("final")
        logger.info("MiniMax H3 LoRA training completed!")

    def save_checkpoint(self, step):
        """Save LoRA checkpoint safetensors file."""
        if not self.is_main_process:
            return

        output_dir = Path(self.config["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f"minimax_h3_lora_step_{step}.safetensors"
        state_dict = self.lora_network.get_state_dict()

        metadata = {
            "step": str(step),
            "rank": str(self.config.get("lora", {}).get("rank", 16)),
            "alpha": str(self.config.get("lora", {}).get("alpha", 16.0)),
            "architecture": "minimax_h3",
        }

        save_file(state_dict, str(save_path), metadata=metadata)
        self.last_saved_path = str(save_path)
        logger.info("Saved LoRA checkpoint to %s", save_path)

    def run(self):
        """Run training pipeline."""
        try:
            set_seed(self.config.get("seed", 42))
            start_time = time.time()

            if self.dry_run:
                self.dry_run_training()
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


def main():
    parser = argparse.ArgumentParser(description="Train MiniMax H3 LoRA in Maltrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without loading models")
    parser.add_argument("--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step during dry run")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}))

    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])

    trainer = MiniMaxH3LoRATrainer(
        config,
        dry_run=args.dry_run,
        dry_run_step_time=args.dry_run_step_time,
    )
    trainer.run()


if __name__ == "__main__":
    main()
