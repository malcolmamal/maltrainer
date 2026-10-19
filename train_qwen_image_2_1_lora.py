#!/usr/bin/env python3
"""
Qwen-Image-2.1 LoRA Training Script for Maltrainer.
Trains LoRA adapters for Qwen-Image-2.1 using INT8 ConvRot transformer, 3D VAE, and Qwen3-VL text encoder.
Uses automated dataset pre-caching for fast step execution and low VRAM footprint.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from maltrainer.data.qwen_dataset import (
    QwenImage21Dataset,
)
from maltrainer.models.qwen_image_2_1.lora import (
    QwenImage21LoRANetwork,
)
from maltrainer.models.qwen_image_2_1.model import (
    apply_flow_shift,
    calculate_shift,
    load_qwen_image_2_1_transformer,
)
from maltrainer.models.qwen_image_2_1.text_encoder import (
    load_qwen_image_2_1_text_encoder,
)
from maltrainer.models.qwen_image_2_1.vae import (
    load_qwen_image_2_1_vae,
)
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.logging_utils import setup_logging
from maltrainer.utils.lora_metadata import build_metadata, git_commit

logger = logging.getLogger(__name__)


def plan_training_steps(num_batches: int, train_config: dict) -> Tuple[int, int, int]:
    """Return (steps_per_epoch, num_epochs, total_steps) for a run.

    ``train.max_steps`` (> 0) wins over ``train.num_epochs``: the run stops exactly at that
    many optimizer steps, and the epoch count is whatever it takes to get there (the last
    epoch may be partial). Otherwise the run is ``num_epochs`` full epochs.
    """
    grad_accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)
    steps_per_epoch = max(math.ceil(num_batches / grad_accum), 1)
    max_steps = int(train_config.get("max_steps", 0) or 0)
    if max_steps > 0:
        return steps_per_epoch, math.ceil(max_steps / steps_per_epoch), max_steps
    num_epochs = max(int(train_config.get("num_epochs", 10) or 1), 1)
    return steps_per_epoch, num_epochs, num_epochs * steps_per_epoch


class QwenImage21LoRATrainer:
    """Trainer for Qwen-Image-2.1 LoRA models."""

    def __init__(self, config: dict, dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run = dry_run
        self.dry_run_step_time = dry_run_step_time
        self.last_saved_path = None
        self.lr_scheduler = None
        self.run_metadata: Dict[str, object] = {
            "model_type": "qwen_image_2_1",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
        }
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
                logger.info("Total VRAM: %.2f GB", torch.cuda.get_device_properties(self.device).total_memory / (1024**3))

        log_dir = Path(config.get("output", {}).get("path", "./output"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        logger.info("Starting dataset pre-caching for Qwen-Image-2.1...")

        temp_dataset = QwenImage21Dataset(
            data_root=dataset_config["path"],
            resolution=dataset_config.get("resolution", 512),
            trigger_token=dataset_config.get("trigger", None),
            default_caption=dataset_config.get("default_caption", ""),
            repeats=1,
        )

        # 1. Load VAE & Encode All Unique Images
        print(f"Loading Qwen 2.1 VAE from {vae_path} for pre-caching...", flush=True)
        vae = load_qwen_image_2_1_vae(vae_path, device=self.device, dtype=self.dtype, include_decoder=False)

        temp_loader = DataLoader(temp_dataset, batch_size=1, shuffle=False)
        print("Caching VAE latents for images...", flush=True)
        for item in tqdm(temp_loader, desc="Caching VAE Latents", disable=not self.is_main_process):
            img_path = item["image_path"][0]
            if img_path not in self.latent_cache:
                pixel_values = item["pixel_values"].to(self.device, dtype=self.dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values, deterministic=True).detach().cpu()
                self.latent_cache[img_path] = latents

        del vae
        gc.collect()
        torch.cuda.empty_cache()
        print("Qwen 2.1 VAE latents pre-cached and VAE unloaded from VRAM.", flush=True)

        # 2. Load Text Encoder & Encode All Unique Captions
        print(f"Loading Qwen3-VL Text Encoder from {text_encoder_path} for pre-caching...", flush=True)
        text_encoder = load_qwen_image_2_1_text_encoder(
            text_encoder_path,
            asset_dir=model_config.get("tokenizer_path"),
            device=self.device,
            dtype=self.dtype,
        )

        unique_captions = list(set(temp_dataset.captions))
        print(f"Caching text embeddings for {len(unique_captions)} unique prompts...", flush=True)
        for caption in tqdm(unique_captions, desc="Caching Prompts", disable=not self.is_main_process):
            if caption not in self.text_cache:
                with torch.no_grad():
                    text_embeds = text_encoder.encode_prompts([caption], device=self.device).detach().cpu()
                self.text_cache[caption] = text_embeds

        # Also cache unconditional prompt if differential guidance is enabled
        train_config = self.config.get("train", {})
        if train_config.get("do_guidance_loss", train_config.get("do_differential_guidance", False)):
            uncond_prompt = str(train_config.get("unconditional_prompt", ""))
            if uncond_prompt not in self.text_cache:
                with torch.no_grad():
                    uncond_embeds = text_encoder.encode_prompts([uncond_prompt], device=self.device).detach().cpu()
                self.text_cache[uncond_prompt] = uncond_embeds

        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        print("Text embeddings pre-cached and Text Encoder unloaded from VRAM.", flush=True)
        logger.info(
            "Pre-cached %d latents and %d prompt embeddings. Unloaded feature extractors from VRAM.",
            len(self.latent_cache),
            len(self.text_cache),
        )

    def load_transformer_and_lora(self):
        """Load INT8 ConvRot / BF16 Qwen-Image-2.1 Transformer Base and inject LoRA network."""
        print("=== Step 2/2: Loading Qwen-Image-2.1 Transformer Base ===", flush=True)
        logger.info("Loading Transformer base and LoRA network...")
        model_config = self.config.get("model", {})
        transformer_path = model_config["transformer_path"]

        grad_precision = self.config.get("train", {}).get("int8_grad_precision", "bf16")
        transformer = load_qwen_image_2_1_transformer(
            transformer_path,
            device=self.device,
            dtype=self.dtype,
            grad_precision=grad_precision,
        )
        self.run_metadata["int8_grad_precision"] = grad_precision
        self.transformer = transformer

        # Inject LoRA
        lora_config = self.config.get("lora", {})
        self.lora_network = QwenImage21LoRANetwork(
            transformer=self.transformer,
            rank=lora_config.get("rank", 16),
            alpha=lora_config.get("alpha", 16.0),
            prefix=lora_config.get("prefix", "diffusion_model."),
            target_modules=lora_config.get("target_modules", None),
        )

        self.lora_dtype = getattr(torch, str(lora_config.get("dtype", "float32")))
        self.lora_network.set_lora_dtype(self.device, self.lora_dtype)

        adapter_path = model_config.get("training_adapter_path") or lora_config.get("training_adapter_path")
        if adapter_path:
            attached = self.lora_network.load_assistant_adapter(
                adapter_path,
                multiplier=float(lora_config.get("training_adapter_multiplier", 1.0)),
                device=self.device,
                dtype=self.dtype,
            )
            self.run_metadata["training_adapter"] = Path(str(adapter_path)).name
            self.run_metadata["training_adapter_modules"] = attached

        trainable = sum(p.numel() for p in self.lora_network.trainable_parameters())
        self.run_metadata["lora_dtype"] = str(self.lora_dtype).replace("torch.", "")
        self.run_metadata["trainable_parameters"] = trainable
        self.run_metadata["lora_layers"] = len(self.lora_network.lora_layers)
        logger.info(
            "LoRA adapters: %.1fM trainable parameters in %s (base left at checkpoint dtypes)",
            trainable / 1e6,
            self.lora_dtype,
        )
        print("Transformer and LoRA network ready on GPU!", flush=True)

    def create_dataset(self):
        """Create image training dataset and dataloader."""
        dataset_config = self.config["dataset"]

        self.train_dataset = QwenImage21Dataset(
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

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        train_config = self.config.get("train", {})
        distribution = str(train_config.get("timestep_distribution", "shift")).lower()
        if distribution in {"sigmoid", "shift", "logit_normal"}:
            scale = float(train_config.get("sigmoid_scale", 1.0))
            z = torch.randn(batch_size, device=self.device, dtype=self.dtype) * scale
            timesteps = torch.sigmoid(z).clamp(0.001, 0.999)
        else:
            timesteps = torch.rand(batch_size, device=self.device, dtype=self.dtype)
        return timesteps

    def setup_optimizer(self):
        """Setup optimizer over the LoRA adapters only."""
        train_config = self.config["train"]
        lr = float(train_config.get("learning_rate", 1e-4))
        optimizer_name = str(train_config.get("optimizer", "adamw8bit")).lower()
        weight_decay = float(train_config.get("weight_decay", 0.01))
        params = list(self.lora_network.trainable_parameters())

        if optimizer_name in {"adamw8bit", "adam8bit"}:
            try:
                import bitsandbytes as bnb

                optimizer_cls = bnb.optim.AdamW8bit if optimizer_name == "adamw8bit" else bnb.optim.Adam8bit
                self.optimizer = optimizer_cls(params, lr=lr, weight_decay=weight_decay)
                logger.info("Using 8-bit optimizer: %s (lr=%.2e)", optimizer_name, lr)
                self.run_metadata["optimizer_resolved"] = {"name": optimizer_name, "lr": lr}
            except Exception as e:
                logger.warning("Bitsandbytes not available (%s). Falling back to AdamW.", e)
                self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.startswith("prodigy"):
            try:
                import inspect

                from prodigyopt import Prodigy

                configured_lr = float(train_config.get("learning_rate", 1.0))
                default_multiplier = configured_lr if configured_lr >= 0.01 else 1.0
                requested = {
                    "lr": float(train_config.get("prodigy_lr", default_multiplier)),
                    "betas": (
                        float(train_config.get("adam_beta1", 0.9)),
                        float(train_config.get("prodigy_beta2", 0.99)),
                    ),
                    "beta3": train_config.get("prodigy_beta3"),
                    "weight_decay": weight_decay,
                    "eps": float(train_config.get("prodigy_eps", 1e-8)),
                    "d0": float(train_config.get("prodigy_d0", 1e-6)),
                    "d_coef": float(train_config.get("prodigy_d_coef", 1.0)),
                    "growth_rate": float(train_config.get("prodigy_growth_rate", float("inf"))),
                    "decouple": bool(train_config.get("prodigy_decouple", True)),
                    "use_bias_correction": bool(train_config.get("prodigy_use_bias_correction", True)),
                    "safeguard_warmup": bool(
                        train_config.get(
                            "prodigy_safeguard_warmup",
                            int(train_config.get("warmup_steps", 0)) > 0,
                        )
                    ),
                }
                sig = inspect.signature(Prodigy.__init__)
                supported = {k: v for k, v in requested.items() if k in sig.parameters and v is not None}
                self.optimizer = Prodigy(params, **supported)
                logger.info(
                    "Using Prodigy optimizer: mult=%.2f, d_coef=%.2f, d0=%.1e",
                    requested["lr"],
                    requested["d_coef"],
                    requested["d0"],
                )
                self.run_metadata["optimizer_resolved"] = {"name": "prodigy", **supported}
            except Exception as e:
                logger.warning("Prodigy not available (%s). Falling back to AdamW.", e)
                self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name in {"automagic3", "automagic", "automagic_v3"}:
            from maltrainer.optimizers import Automagic3

            configured_lr = float(train_config.get("learning_rate", 1e-6))
            automagic_params = {
                "lr": configured_lr,
                "min_lr": float(train_config.get("automagic_min_lr", train_config.get("min_lr", 1e-8))),
                "max_lr": float(train_config.get("automagic_max_lr", train_config.get("max_lr", 1e3))),
                "beta2": float(train_config.get("automagic_beta2", train_config.get("beta2", 0.999))),
                "eps": float(train_config.get("automagic_eps", train_config.get("eps", 1e-30))),
                "clip_threshold": float(train_config.get("automagic_clip_threshold", train_config.get("clip_threshold", 1.0))),
                "weight_decay": weight_decay,
                "polarity_history": int(train_config.get("automagic_polarity_history", train_config.get("polarity_history", 8))),
                "fused": bool(train_config.get("automagic_fused", False)),
            }
            self.optimizer = Automagic3(params, **automagic_params)
            logger.info(
                "Using Automagic3 optimizer: start_lr=%.2e, history=%d, clip_threshold=%.2f",
                automagic_params["lr"],
                automagic_params["polarity_history"],
                automagic_params["clip_threshold"],
            )
            self.run_metadata["optimizer_resolved"] = {"name": "automagic3", **automagic_params}
        else:
            self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
            self.run_metadata["optimizer_resolved"] = {"name": "adamw", "lr": lr}

        warmup_steps = int(train_config.get("warmup_steps", 0))
        if warmup_steps > 0:
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(current_step: int):
                return min(1.0, float(current_step) / float(max(1, warmup_steps)))

            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
            logger.info("Configured learning rate warmup for %d steps.", warmup_steps)

    def train(self):
        """Main Qwen-Image-2.1 LoRA training loop."""
        train_config = self.config["train"]
        save_every = train_config.get("save_every", 500)

        self.lora_network, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.lora_network, self.optimizer, self.train_dataloader
        )

        global_step = 0
        steps_per_epoch, num_epochs, total_steps = plan_training_steps(len(self.train_dataloader), train_config)
        self.run_metadata["steps_per_epoch"] = steps_per_epoch
        self.run_metadata["planned_total_steps"] = total_steps
        self.run_metadata["dataset_samples"] = len(self.train_dataset)

        logger.info(
            "Starting Qwen-Image-2.1 LoRA training for %d epochs (%d total steps)...",
            num_epochs,
            total_steps,
        )

        progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not self.is_main_process)

        for epoch in range(num_epochs):
            if global_step >= total_steps:
                break
            self.lora_network.train()

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.lora_network):
                    image_paths = batch["image_path"]
                    captions = batch["caption"]

                    # 1. Fetch pre-cached latents [B, 64, H_lat, W_lat]
                    latents = torch.cat([self.latent_cache[p] for p in image_paths], dim=0).to(
                        device=self.device, dtype=self.dtype
                    )

                    # 2. Fetch pre-cached text embeddings [B, text_len, 4096]
                    text_embeds = torch.cat([self.text_cache[c] for c in captions], dim=0).to(
                        device=self.device, dtype=self.dtype
                    )

                    bsz, c_lat, h_lat, w_lat = latents.shape
                    img_tokens = h_lat * w_lat

                    # 3. Dynamic shift and timestep sampling
                    mu_shift = calculate_shift(img_tokens)
                    raw_timesteps = self._sample_timesteps(bsz)
                    shifted_sigmas = apply_flow_shift(raw_timesteps, mu=mu_shift)

                    # 4. Sample noise and create noisy latents
                    # Flow matching interpolation: x_t = (1 - t) * x_0 + t * noise
                    noise = torch.randn_like(latents)
                    sigma_broadcast = shifted_sigmas.view(bsz, 1, 1, 1)
                    noisy_latents = (1.0 - sigma_broadcast) * latents + sigma_broadcast * noise

                    # 5. Model forward pass
                    # Target velocity: v = noise - x_0
                    target_velocity = noise - latents

                    # Flatten latents: [B, 64, H, W] -> [B, H*W, 64]
                    noisy_flat = noisy_latents.permute(0, 2, 3, 1).reshape(bsz, img_tokens, c_lat)
                    target_flat = target_velocity.permute(0, 2, 3, 1).reshape(bsz, img_tokens, c_lat)

                    pred_velocity = self.lora_network(
                        hidden_states=noisy_flat,
                        encoder_hidden_states=text_embeds,
                        timestep=shifted_sigmas,
                        latent_h=h_lat,
                        latent_w=w_lat,
                    )

                    loss = torch.nn.functional.mse_loss(pred_velocity.float(), target_flat.float())

                    loss_weight_strength = float(train_config.get("loss_weight_strength", 1.0))
                    if loss_weight_strength != 1.0:
                        loss = loss * loss_weight_strength

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.lora_network.trainable_parameters(),
                            train_config.get("max_grad_norm", 1.0),
                        )
                        self.optimizer.step()
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    group = self.optimizer.param_groups[0]
                    current_lr = group["lr"]
                    prodigy_d = group.get("d")
                    # accelerate's AcceleratedOptimizer does not forward custom methods
                    inner_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
                    if hasattr(inner_optimizer, "get_avg_learning_rate"):
                        adapted_lr = inner_optimizer.get_avg_learning_rate()
                        self.progress_logger.info(
                            "Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, LR: %.2e (automagic3)",
                            global_step,
                            total_steps,
                            int(global_step / total_steps * 100),
                            loss.item(),
                            epoch + 1,
                            num_epochs,
                            adapted_lr,
                        )
                    elif prodigy_d is not None:
                        self.progress_logger.info(
                            "Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, LR: %.2e (mult %.3f x d %.2e)",
                            global_step,
                            total_steps,
                            int(global_step / total_steps * 100),
                            loss.item(),
                            epoch + 1,
                            num_epochs,
                            current_lr * prodigy_d,
                            current_lr,
                            prodigy_d,
                        )
                    else:
                        self.progress_logger.info(
                            "Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, LR: %.2e",
                            global_step,
                            total_steps,
                            int(global_step / total_steps * 100),
                            loss.item(),
                            epoch + 1,
                            num_epochs,
                            current_lr,
                        )

                    if global_step % save_every == 0 and global_step < total_steps:
                        self.save_checkpoint(global_step)

                progress_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "step": global_step, "epoch": f"{epoch + 1}/{num_epochs}"}
                )
                if global_step >= total_steps:
                    break

        progress_bar.close()
        self.save_checkpoint("final")
        logger.info("Qwen-Image-2.1 LoRA training completed!")

    def save_checkpoint(self, step):
        """Save LoRA checkpoint safetensors file."""
        if not self.is_main_process:
            return

        output_dir = Path(self.config["output"]["path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        output_config = self.config.get("output", {})
        lora_name = (
            output_config.get("name")
            or output_config.get("lora_name")
            or output_config.get("file_name")
            or "qwen_image_2_1_lora"
        )
        if lora_name.endswith(".safetensors"):
            lora_name = lora_name[:-len(".safetensors")]

        if step == "final":
            save_path = output_dir / f"{lora_name}.safetensors"
        else:
            save_path = output_dir / f"{lora_name}_step_{step}.safetensors"

        save_dtype = getattr(torch, str(self.config.get("output", {}).get("dtype", "bfloat16")))
        format_type = str(self.config.get("output", {}).get("format", "diffusers")).lower()
        state_dict = self.lora_network.get_state_dict(save_dtype=save_dtype, format_type=format_type)

        runtime = dict(self.run_metadata)
        runtime["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        runtime["save_dtype"] = str(save_dtype).replace("torch.", "")
        metadata = build_metadata(
            self.config,
            step=step,
            runtime=runtime,
            embed_config=bool(self.config.get("output", {}).get("embed_config", True)),
        )

        save_file(state_dict, str(save_path), metadata=metadata)
        self.last_saved_path = str(save_path)
        logger.info("Saved LoRA checkpoint to %s", save_path)

    def simulate(self):
        """Dry run: plan the run from the real dataset, then tick through the steps without models.

        Emits the same progress bar and progress-log lines as a real run, so the web UI's
        queueing and progress tracking can be exercised without a GPU.
        """
        self.create_dataset()
        train_config = self.config["train"]
        steps_per_epoch, num_epochs, total_steps = plan_training_steps(len(self.train_dataloader), train_config)
        save_every = int(train_config.get("save_every", 500))
        logger.info(
            "Dry run: %d samples, %d steps/epoch -> %d epochs, %d total steps (%.3fs per simulated step)",
            len(self.train_dataset), steps_per_epoch, num_epochs, total_steps, self.dry_run_step_time,
        )
        with tqdm(total=total_steps, desc="Training Steps (dry run)", disable=not self.is_main_process) as bar:
            for step in range(1, total_steps + 1):
                time.sleep(self.dry_run_step_time)
                bar.update(1)
                epoch = math.ceil(step / steps_per_epoch)
                self.progress_logger.info(
                    "Step: %d/%d [%d%%], Loss: 0.000000, Epoch: %d/%d, LR: 0.00e+00 (dry run)",
                    step, total_steps, int(step / total_steps * 100), epoch, num_epochs,
                )
                if step % save_every == 0 and step < total_steps:
                    logger.info("Dry run: would save checkpoint at step %d", step)
        logger.info("Dry run complete: would save final checkpoint at step %d", total_steps)

    def run(self):
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


def main():
    parser = argparse.ArgumentParser(description="Train Qwen-Image-2.1 LoRA in Maltrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without loading models")
    parser.add_argument("--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step in a dry run")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}))

    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])

    trainer = QwenImage21LoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time)
    trainer.run()


if __name__ == "__main__":
    main()
