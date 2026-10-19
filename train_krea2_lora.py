#!/usr/bin/env python3
"""
Krea 2 LoRA training for maltrainer.

Trains on ComfyUI's single files: the RAW DiT (fp8_scaled, or bf16 quantized to fp8 on
load), the Qwen-Image / Wan 2.1 VAE and the Qwen3-VL-4B text encoder. Latents and caption
embeddings are cached once, then the encoders are unloaded before the DiT loads.

Conventions follow ComfyUI's Krea 2 implementation (comfy/ldm/krea2, ModelType.FLUX): the DiT
gets sigma in [0, 1] and predicts v = noise - x0.

    python train_krea2_lora.py --config templates/krea2_template.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.optim import AdamW
from tqdm import tqdm

from maltrainer.data.krea2_dataset import load_krea2_image
from maltrainer.data.minimax_dataset import count_media_files, scan_media_sources
from maltrainer.models.krea2.loading import load_krea2_transformer
from maltrainer.models.krea2.lora import Krea2LoRANetwork
from maltrainer.models.krea2.text_encoder import load_krea2_text_encoder
from maltrainer.models.krea2.vae import load_krea2_vae
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.ema import ParameterEMA
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.logging_utils import setup_logging
from maltrainer.utils.lora_metadata import build_metadata, git_commit

logger = logging.getLogger(__name__)


def plan_training_steps(num_samples: int, train_config: dict) -> Tuple[int, int, int]:
    """(steps_per_epoch, num_epochs, total_steps); ``train.max_steps`` > 0 wins over epochs."""
    grad_accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)
    steps_per_epoch = max(math.ceil(num_samples / grad_accum), 1)
    max_steps = int(train_config.get("max_steps", 0) or 0)
    if max_steps > 0:
        return steps_per_epoch, math.ceil(max_steps / steps_per_epoch), max_steps
    num_epochs = max(int(train_config.get("num_epochs", 10) or 1), 1)
    return steps_per_epoch, num_epochs, num_epochs * steps_per_epoch


def krea2_shift(image_tokens: int) -> float:
    """Resolution-dependent sigma shift (musubi-tuner / Fizgig): mu 0.5 at 256 tokens, 1.15 at 6400."""
    mu = 0.5 + (image_tokens - 256) * (1.15 - 0.5) / (6400 - 256)
    return math.exp(mu)


class Krea2LoRATrainer:
    def __init__(self, config: dict, dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run = dry_run
        self.dry_run_step_time = dry_run_step_time
        self.last_saved_path: Optional[str] = None
        self.lr_scheduler = None
        self.ema: Optional[ParameterEMA] = None
        self.run_metadata: Dict[str, object] = {
            "model_type": "krea2",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.samples: List[Tuple[str, str]] = []  # (key, caption)
        self.latent_cache: Dict[str, torch.Tensor] = {}
        self.text_cache: Dict[str, torch.Tensor] = {}
        self.dop_captions: Dict[str, str] = {}  # sample key -> its caption with the trigger replaced
        self.micro_step = 0
        self.last_dop_loss: Optional[float] = None

        train_config = config.get("train", {})
        self.dop = bool(train_config.get("dop", False))
        if self.dop:
            if not str(train_config.get("dop_class", "") or "").strip():
                raise ValueError("train.dop needs train.dop_class, the class word that replaces the trigger")
            if not str(config.get("dataset", {}).get("trigger", "") or "").strip():
                raise ValueError("train.dop needs dataset.trigger: DOP's captions are the captions without it")
        if int(train_config.get("batch_size", 1) or 1) != 1:
            raise ValueError(
                "Krea 2 trains at batch_size 1 (every caption keeps its own unpadded length, as in "
                "ComfyUI); use train.gradient_accumulation_steps for a larger effective batch."
            )

        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        )
        self.is_main_process = self.accelerator.is_main_process
        if self.is_main_process:
            name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
            logger.info("Training on device: %s (%s)", self.device, name)
            if self.device.type == "cuda":
                logger.info("Total VRAM: %.2f GB", torch.cuda.get_device_properties(self.device).total_memory / 1024**3)

        log_dir = Path(config.get("output", {}).get("path", "./output"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)
        self.progress_logger = logging.getLogger("progress")
        self.progress_logger.setLevel(logging.INFO)
        self.progress_logger.addHandler(file_handler)
        self.progress_logger.propagate = False

    # ------------------------------------------------------------------ caching

    def _sources(self):
        dataset_config = self.config["dataset"]
        return scan_media_sources(
            dataset_config["path"],
            kinds=["image"],
            trigger_token=dataset_config.get("trigger"),
            default_caption=dataset_config.get("default_caption", ""),
        )

    def precache_dataset(self):
        model_config = self.config["model"]
        dataset_config = self.config["dataset"]
        train_config = self.config.get("train", {})
        sources = self._sources()
        if not sources:
            raise ValueError(f"No images found under {dataset_config['path']}")

        print("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text Embeddings) ===", flush=True)
        resolution = int(dataset_config.get("resolution", 512))
        aspect_mode = str(dataset_config.get("aspect_mode", "bucket")).lower()

        vae = load_krea2_vae(model_config["vae_path"], self.device)
        for source in tqdm(sources, desc="Caching latents", disable=not self.is_main_process):
            try:
                pixels = load_krea2_image(source.path, resolution, aspect_mode).unsqueeze(0).to(self.device)
                self.latent_cache[str(source.path)] = vae.encode(pixels).cpu()
                self.samples.append((str(source.path), source.caption))
            except Exception as e:
                logger.warning("Skipping %s: %s", source.path.name, e)
        del vae
        gc.collect()
        torch.cuda.empty_cache()
        print("VAE latents pre-cached and VAE unloaded from VRAM.", flush=True)

        prompts = {caption for _, caption in self.samples}
        if float(train_config.get("caption_dropout", 0.0) or 0.0) > 0:
            prompts.add("")
        if self.dop:
            self._plan_dop_captions()
            prompts.update(self.dop_captions.values())
        encoder = load_krea2_text_encoder(
            model_config["text_encoder_path"], self.device, self.dtype, model_config.get("tokenizer_path")
        )
        for caption in tqdm(sorted(prompts), desc="Caching prompts", disable=not self.is_main_process):
            self.text_cache[caption] = encoder.encode(caption, self.device, self.dtype).cpu()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        print("Text embeddings pre-cached and Text Encoder unloaded from VRAM.", flush=True)
        logger.info("Pre-cached %d images and %d prompt embeddings.", len(self.samples), len(self.text_cache))

    def _plan_dop_captions(self) -> None:
        """DOP's caption per sample: the trigger replaced by the class word, as ai-toolkit does."""
        train_config = self.config["train"]
        trigger = str(self.config["dataset"]["trigger"]).strip()
        cls = str(train_config["dop_class"]).strip()
        for key, caption in self.samples:
            if trigger in caption:
                self.dop_captions[key] = caption.replace(trigger, cls)
        missing = len(self.samples) - len(self.dop_captions)
        example = next(((c, self.dop_captions[k]) for k, c in self.samples if k in self.dop_captions), None)
        logger.info(
            "DOP: %d of %d captions name the trigger %r; their DOP caption swaps it for %r%s",
            len(self.dop_captions), len(self.samples), trigger, cls,
            f" (e.g. {example[0]!r} -> {example[1]!r})" if example else "",
        )
        if missing:
            logger.warning(
                "DOP: %d caption(s) don't contain the trigger %r and get no DOP pass (holding them to the "
                "base model would unlearn the subject).", missing, trigger,
            )
        if not self.dop_captions:
            raise ValueError(f"train.dop: no caption contains the trigger {trigger!r}")
        self.run_metadata["dop"] = {
            "class": cls,
            "multiplier": float(train_config.get("dop_multiplier", 1.0)),
            "every": int(train_config.get("dop_every", 1) or 1),
            "captions": len(self.dop_captions),
        }

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self):
        print("=== Step 2/2: Loading Krea 2 DiT ===", flush=True)
        model_config = self.config["model"]
        lora_config = self.config.get("lora", {})
        self.transformer, load_info = load_krea2_transformer(model_config["transformer_path"], self.device, self.dtype)
        self.run_metadata["base_load"] = load_info
        self.transformer.enable_gradient_checkpointing()

        self.lora_network = Krea2LoRANetwork(
            self.transformer,
            rank=int(lora_config.get("rank", 16)),
            alpha=float(lora_config.get("alpha", lora_config.get("rank", 16))),
            targets=str(lora_config.get("targets", "attn-mlp")),
            block_range=lora_config.get("block_range"),
            prefix=lora_config.get("prefix", "diffusion_model."),
        )
        self.lora_dtype = getattr(torch, str(lora_config.get("dtype", "float32")))
        self.lora_network.set_lora_dtype(self.device, self.lora_dtype)
        trainable = sum(p.numel() for p in self.lora_network.trainable_parameters())
        self.run_metadata.update({
            "lora_layers": len(self.lora_network.lora_layers),
            "trainable_parameters": trainable,
            "lora_dtype": str(self.lora_dtype).replace("torch.", ""),
        })
        logger.info("LoRA: %.1fM trainable parameters in %s", trainable / 1e6, self.lora_dtype)

    def setup_optimizer(self):
        train_config = self.config["train"]
        name = str(train_config.get("optimizer", "automagic3")).lower()
        weight_decay = float(train_config.get("weight_decay", 0.0))
        lr = float(train_config.get("learning_rate", 1e-6))

        if name in {"automagic3", "automagic", "automagic_v3"}:
            from maltrainer.optimizers import Automagic3

            # Fizgig: each module family (attention, MLP, text fusion, I/O) adapts its own rate,
            # and a 16-step sign window, because at 8 the rate "crawled from 1e-6 to 4.3e-6
            # across a whole epoch" on batch-1 Krea 2 gradients.
            groups = self.lora_network.param_groups(bool(train_config.get("automagic_param_groups", True)))
            kwargs = {
                "lr": lr,
                "min_lr": float(train_config.get("automagic_min_lr", 1e-8)),
                "max_lr": float(train_config.get("automagic_max_lr", 1e3)),
                "beta2": float(train_config.get("automagic_beta2", 0.999)),
                "eps": float(train_config.get("automagic_eps", 1e-30)),
                "clip_threshold": float(train_config.get("automagic_clip_threshold", 1.0)),
                "weight_decay": weight_decay,
                "polarity_history": int(train_config.get("automagic_polarity_history", 16)),
                "fused": bool(train_config.get("automagic_fused", False)),
            }
            self.optimizer = Automagic3(groups, **kwargs)
            self.run_metadata["optimizer_resolved"] = {"name": "automagic3", "groups": [g["name"] for g in groups], **kwargs}
        elif name.startswith("prodigy"):
            import inspect

            from prodigyopt import Prodigy

            requested = {
                "lr": float(train_config.get("prodigy_lr", lr if lr >= 0.01 else 1.0)),
                "betas": (float(train_config.get("adam_beta1", 0.9)), float(train_config.get("prodigy_beta2", 0.999))),
                "beta3": train_config.get("prodigy_beta3"),
                "weight_decay": weight_decay,
                "eps": float(train_config.get("prodigy_eps", 1e-8)),
                "d0": float(train_config.get("prodigy_d0", 1e-6)),
                "d_coef": float(train_config.get("prodigy_d_coef", 1.0)),
                "growth_rate": float(train_config.get("prodigy_growth_rate", float("inf"))),
                "decouple": bool(train_config.get("prodigy_decouple", True)),
                "use_bias_correction": bool(train_config.get("prodigy_use_bias_correction", False)),
                "safeguard_warmup": bool(train_config.get("prodigy_safeguard_warmup", False)),
                "slice_p": int(train_config.get("prodigy_slice_p", 1)),
            }
            supported = set(inspect.signature(Prodigy.__init__).parameters)
            kwargs = {k: v for k, v in requested.items() if k in supported and v is not None}
            dropped = sorted(set(requested) - set(kwargs) - {"beta3"})
            if dropped:
                logger.warning("Installed prodigyopt does not accept %s; ignored.", dropped)
            self.optimizer = Prodigy(self.lora_network.trainable_parameters(), **kwargs)
            self.run_metadata["optimizer_resolved"] = {"name": "prodigy", **kwargs}
        elif name in {"adamw8bit", "adam8bit"}:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(self.lora_network.trainable_parameters(), lr=lr, weight_decay=weight_decay)
            self.run_metadata["optimizer_resolved"] = {"name": name, "lr": lr}
        else:
            self.optimizer = AdamW(self.lora_network.trainable_parameters(), lr=lr, weight_decay=weight_decay)
            self.run_metadata["optimizer_resolved"] = {"name": "adamw", "lr": lr}
        logger.info("Optimizer: %s", self.run_metadata["optimizer_resolved"])

        warmup = int(train_config.get("learning_rate_warmup_steps", 0) or 0)
        if warmup > 0:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min((step + 1) / warmup, 1.0))
            logger.info("Learning-rate warmup over %d steps.", warmup)

    # ------------------------------------------------------------------ training

    def _sample_sigma(self, image_tokens: int) -> Tuple[torch.Tensor, float]:
        """Video-free flow-matching sigma for one sample, plus the shift used."""
        train_config = self.config["train"]
        distribution = str(train_config.get("timestep_distribution", "krea2_shift")).lower()
        if distribution in {"logit_normal", "logit-normal", "sigmoid", "krea2_shift"}:
            mean = float(train_config.get("timestep_logit_mean", 0.0))
            std = float(train_config.get("timestep_logit_std", 1.0))
            base = torch.sigmoid(torch.randn(1, device=self.device) * std + mean)
        else:  # uniform
            base = torch.rand(1, device=self.device)
        if distribution == "krea2_shift":
            shift = krea2_shift(image_tokens)
        elif distribution == "sigmoid":
            shift = 1.0
        else:
            shift = float(train_config.get("timestep_shift", 1.0))
        sigma = shift * base / (1.0 + (shift - 1.0) * base)
        lo = float(train_config.get("min_sigma", 0.001))
        hi = float(train_config.get("max_sigma", 0.999))
        return sigma.clamp(lo, hi), shift

    def _epoch_order(self) -> List[int]:
        repeats = max(int(self.config["dataset"].get("repeats", 1) or 1), 1)
        order = [i for i in range(len(self.samples)) for _ in range(repeats)]
        random.shuffle(order)
        world = self.accelerator.num_processes
        return order[self.accelerator.process_index::world] if world > 1 else order

    def _training_step(self, index: int, shift_logged: List[bool]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The flow-matching loss for one sample, and the noised latent and sigma it used."""
        train_config = self.config["train"]
        key, caption = self.samples[index]
        dropout = float(train_config.get("caption_dropout", 0.0) or 0.0)
        if dropout > 0 and torch.rand(()).item() < dropout:
            caption = ""
        x0 = self.latent_cache[key].to(self.device, torch.float32)
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)

        tokens = (x0.shape[-2] // 2) * (x0.shape[-1] // 2)
        sigma, shift = self._sample_sigma(tokens)
        if not shift_logged[0]:
            shift_logged[0] = True
            self.run_metadata["resolved_sigma_shift"] = round(shift, 6)
            logger.info(
                "Timestep schedule: %s, sigma shift %.3f at %d image tokens",
                train_config.get("timestep_distribution", "krea2_shift"), shift, tokens,
            )

        noise = torch.randn_like(x0)
        s = sigma.view(1, 1, 1, 1)
        noisy = ((1.0 - s) * x0 + s * noise).to(self.dtype)
        pred = self.lora_network(noisy, sigma, context)
        loss = torch.nn.functional.mse_loss(pred.float(), (noise - x0).float())
        strength = float(train_config.get("loss_weight_strength", 1.0))
        return (loss * strength if strength != 1.0 else loss), noisy, sigma

    def _dop_loss(self, index: int, noisy: torch.Tensor, sigma: torch.Tensor) -> Optional[torch.Tensor]:
        """Differential output preservation (ai-toolkit's diff_output_preservation).

        On the same noised latent and sigma, the caption without the trigger: the base model's
        prediction (adapters off, no grad) is the target for the LoRA's prediction. None when this
        sample's caption has no trigger, or it isn't a DOP step (``train.dop_every``).
        """
        train_config = self.config["train"]
        caption = self.dop_captions.get(self.samples[index][0])
        every = max(int(train_config.get("dop_every", 1) or 1), 1)
        if caption is None or self.micro_step % every:
            return None
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)
        with torch.no_grad(), self.accelerator.unwrap_model(self.lora_network).disabled():
            target = self.lora_network(noisy, sigma, context).float()
        pred = self.lora_network(noisy, sigma, context)
        return torch.nn.functional.mse_loss(pred.float(), target) * float(train_config.get("dop_multiplier", 1.0))

    def train(self):
        train_config = self.config["train"]
        save_every = int(train_config.get("save_every", 250) or 0)
        self.lora_network, self.optimizer = self.accelerator.prepare(self.lora_network, self.optimizer)
        network = self.accelerator.unwrap_model(self.lora_network)

        ema_decay = float(train_config.get("ema_decay", 0.0) or 0.0)
        if ema_decay > 0:
            self.ema = ParameterEMA(network.trainable_parameters(), decay=ema_decay, warmup=bool(train_config.get("ema_warmup", True)))
            self.run_metadata["ema_decay"] = ema_decay
            logger.info("EMA of the LoRA weights: decay %.4f (checkpoints save the average)", ema_decay)

        steps_per_epoch, num_epochs, total_steps = plan_training_steps(len(self._epoch_order()), train_config)
        self.run_metadata.update({"steps_per_epoch": steps_per_epoch, "planned_total_steps": total_steps, "dataset_samples": len(self.samples)})
        logger.info("Starting Krea 2 LoRA training: %d epochs, %d total steps", num_epochs, total_steps)

        progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not self.is_main_process)
        keys = KeyCommands() if self.is_main_process else None
        stopped = False
        global_step = 0
        shift_logged = [False]
        loss = None
        for epoch in range(num_epochs):
            self.lora_network.train()
            for index in self._epoch_order():
                with self.accelerator.accumulate(self.lora_network):
                    loss, noisy, sigma = self._training_step(index, shift_logged)
                    self.accelerator.backward(loss)
                    if self.dop:
                        # its own backward, so the first pass's graph is freed before the second
                        dop_loss = self._dop_loss(index, noisy, sigma)
                        if dop_loss is not None:
                            self.accelerator.backward(dop_loss)
                            self.last_dop_loss = dop_loss.item()
                    self.micro_step += 1
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(network.trainable_parameters(), float(train_config.get("max_grad_norm", 1.0)))
                        self.optimizer.step()
                        if self.lr_scheduler is not None:
                            self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        if self.ema is not None:
                            self.ema.update()

                if self.accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)
                    self._log_step(global_step, total_steps, loss, epoch, num_epochs)
                    if save_every > 0 and global_step % save_every == 0 and global_step < total_steps:
                        self.save_checkpoint(global_step)
                    stopped = keys is not None and keys.handle(global_step, self.save_checkpoint)
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step, "epoch": f"{epoch + 1}/{num_epochs}"})
                if global_step >= total_steps or stopped:
                    break
            if global_step >= total_steps or stopped:
                break

        progress_bar.close()
        if keys is not None:
            keys.close()
        if stopped:
            logger.info("Krea 2 LoRA training stopped by key at step %d (saved as step %d).", global_step, global_step)
            return
        self.save_checkpoint("final")
        logger.info("Krea 2 LoRA training completed!")

    def _log_step(self, step, total, loss, epoch, num_epochs):
        group = self.optimizer.param_groups[0]
        inner = getattr(self.optimizer, "optimizer", self.optimizer)
        prefix = ("Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, " % (step, total, int(step / total * 100), loss.item(), epoch + 1, num_epochs))
        if self.last_dop_loss is not None:
            prefix += "DOP: %.6f, " % self.last_dop_loss
        if hasattr(inner, "get_avg_learning_rate"):
            rates = ", ".join(f"{g.get('name', i)} {lr:.2e}" for i, (g, lr) in enumerate(zip(inner.param_groups, inner.get_learning_rates())))
            self.progress_logger.info("%sLR: %.2e (automagic3: %s)", prefix, inner.get_avg_learning_rate(), rates)
        elif group.get("d") is not None:
            self.progress_logger.info("%sLR: %.2e (mult %.3f x d %.2e)", prefix, group["lr"] * group["d"], group["lr"], group["d"])
        else:
            self.progress_logger.info("%sLR: %.2e", prefix, group["lr"])

    def save_checkpoint(self, step):
        if not self.is_main_process:
            return
        output_config = self.config.get("output", {})
        output_dir = Path(output_config["path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        name = output_config.get("name") or output_config.get("lora_name") or "krea2_lora"
        name = name[: -len(".safetensors")] if name.endswith(".safetensors") else name
        save_path = output_dir / (f"{name}.safetensors" if step == "final" else f"{name}_step_{step}.safetensors")
        save_dtype = getattr(torch, str(output_config.get("dtype", "bfloat16")))
        network = self.accelerator.unwrap_model(self.lora_network)
        if self.ema is not None:
            with self.ema.applied():
                state_dict = network.get_state_dict(save_dtype)
        else:
            state_dict = network.get_state_dict(save_dtype)
        runtime = dict(self.run_metadata, saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"), save_dtype=str(save_dtype).replace("torch.", ""))
        metadata = build_metadata(self.config, step=step, runtime=runtime, embed_config=bool(output_config.get("embed_config", True)))
        save_file(state_dict, str(save_path), metadata=metadata)
        self.last_saved_path = str(save_path)
        logger.info("Saved LoRA checkpoint to %s", save_path)

    # ------------------------------------------------------------------ dry run / entry

    def simulate(self):
        """Plan from the real dataset folder and tick through the steps without loading models."""
        dataset_config = self.config["dataset"]
        repeats = max(int(dataset_config.get("repeats", 1) or 1), 1)
        samples = max(count_media_files(dataset_config.get("path", ""), ["image"]), 1) * repeats
        steps_per_epoch, num_epochs, total_steps = plan_training_steps(samples, self.config["train"])
        logger.info("Dry run: %d samples, %d steps/epoch -> %d epochs, %d total steps", samples, steps_per_epoch, num_epochs, total_steps)
        with tqdm(total=total_steps, desc="Training Steps (dry run)", disable=not self.is_main_process) as bar:
            for step in range(1, total_steps + 1):
                time.sleep(self.dry_run_step_time)
                bar.update(1)
                self.progress_logger.info(
                    "Step: %d/%d [%d%%], Loss: 0.000000, Epoch: %d/%d, LR: 0.00e+00 (dry run)",
                    step, total_steps, int(step / total_steps * 100), math.ceil(step / steps_per_epoch), num_epochs,
                )
        logger.info("Dry run complete: would save final checkpoint at step %d", total_steps)

    def run(self):
        try:
            set_seed(self.config.get("seed", 42))
            if self.dry_run:
                self.simulate()
                return
            self.precache_dataset()
            self.load_transformer_and_lora()
            self.setup_optimizer()
            self.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Exiting...")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error("An error occurred during training: %s", e, exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description="Train a Krea 2 LoRA in maltrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without loading models")
    parser.add_argument("--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}))
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])
    Krea2LoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
