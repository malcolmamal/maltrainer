#!/usr/bin/env python3
"""
FLUX.3 [klein] LoRA training for maltrainer.

Trains the base (undistilled) Klein on ComfyUI's single files: the DiT
(flux3_image_klein_int8_convrot, or _bf16), the Flux2 VAE and the Qwen3-VL-4B text encoder
(qwen3vl_4b_int8_convrot, or _bf16). Latents and caption embeddings are cached once, then the
encoders are unloaded before the DiT loads. The prompt upsampler is an inference-time tool and
is not used.

Conventions follow ComfyUI's FLUX.3 (comfy/ldm/flux3, ModelType.FLUX) and the diffusers FLUX.3
trainer. The DiT gets sigma in [0, 1] and predicts v = noise - x0. Sigmas come from the
reference schedule's fixed shift of 6.93, with no resolution-dependent shift.

    python train_flux3_klein_lora.py --config templates/flux3_klein_template.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.optim import AdamW
from tqdm import tqdm

from maltrainer.data.flux3_dataset import load_flux3_image
from maltrainer.data.minimax_dataset import MediaSource, count_media_files, scan_media_sources
from maltrainer.models.flux3.loading import load_flux3_transformer
from maltrainer.models.flux3.lora import Flux3LoRANetwork
from maltrainer.models.flux3.text_encoder import load_flux3_text_encoder
from maltrainer.models.flux3.vae import load_flux2_vae_encoder
from maltrainer.optimizers.prodigy import build_prodigy
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.ema import ParameterEMA
from maltrainer.utils.logging_utils import setup_logging
from maltrainer.utils.lora_metadata import build_metadata, git_commit

logger = logging.getLogger(__name__)

# The FLUX.3 reference sampler's shift (ComfyUI sampling_settings mu = ln 6.93, the diffusers
# pipeline's Flux3SetTimestepsStep default). The same at every resolution.
FLUX3_SHIFT = 6.93


def plan_training_steps(num_samples: int, train_config: dict[str, Any]) -> tuple[int, int, int]:
    """(steps_per_epoch, num_epochs, total_steps); max_steps wins. An epoch yields num_samples / grad_accum steps."""
    grad_accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)
    steps_per_epoch = max(math.ceil(num_samples / grad_accum), 1)
    max_steps = int(train_config.get("max_steps", 0) or 0)
    if max_steps > 0:
        return steps_per_epoch, math.ceil(max_steps * grad_accum / max(num_samples, 1)), max_steps
    num_epochs = max(int(train_config.get("num_epochs", 10) or 1), 1)
    return steps_per_epoch, num_epochs, max(num_epochs * max(num_samples, 1) // grad_accum, 1)


class Flux3KleinLoRATrainer:
    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run = dry_run
        self.dry_run_step_time = dry_run_step_time
        self.last_saved_path: str | None = None
        self.lr_scheduler: Any = None
        self.ema: ParameterEMA | None = None
        self.run_metadata: dict[str, object] = {
            "model_type": "flux3_klein",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.samples: list[tuple[str, str]] = []  # (key, caption)
        self.latent_cache: dict[str, torch.Tensor] = {}
        self.text_cache: dict[str, torch.Tensor] = {}

        train_config = config.get("train", {})
        if int(train_config.get("batch_size", 1) or 1) != 1:
            raise ValueError(
                "FLUX.3 trains at batch_size 1 (every caption keeps its own 80-token text window, "
                "as at inference); use train.gradient_accumulation_steps for a larger effective batch."
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

    def _sources(self) -> list[MediaSource]:
        dataset_config = self.config["dataset"]
        return scan_media_sources(
            dataset_config["path"],
            kinds=["image"],
            trigger_token=dataset_config.get("trigger"),
            default_caption=dataset_config.get("default_caption", ""),
        )

    def precache_dataset(self) -> None:
        model_config = self.config["model"]
        dataset_config = self.config["dataset"]
        train_config = self.config.get("train", {})
        sources = self._sources()
        if not sources:
            raise ValueError(f"No images found under {dataset_config['path']}")

        print("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text Embeddings) ===", flush=True)
        resolution = int(dataset_config.get("resolution", 512))
        aspect_mode = str(dataset_config.get("aspect_mode", "bucket")).lower()

        vae = load_flux2_vae_encoder(model_config["vae_path"], self.device)
        sizes: dict[tuple[int, int], int] = {}
        for source in tqdm(sources, desc="Caching latents", disable=not self.is_main_process):
            try:
                pixels = load_flux3_image(source.path, resolution, aspect_mode).unsqueeze(0).to(self.device)
                latent = vae.encode(pixels)[0].cpu()
                self.latent_cache[str(source.path)] = latent
                self.samples.append((str(source.path), source.caption))
                size = (pixels.shape[-1], pixels.shape[-2])
                sizes[size] = sizes.get(size, 0) + 1
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Skipping %s: %s", source.path.name, e)
        del vae
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(
            "Training sizes (w x h: images): %s", ", ".join(f"{w}x{h}: {n}" for (w, h), n in sorted(sizes.items()))
        )
        print("VAE latents pre-cached and VAE unloaded from VRAM.", flush=True)

        prompts = {caption for _, caption in self.samples}
        if float(train_config.get("caption_dropout", 0.0) or 0.0) > 0:
            prompts.add("")
        encoder = load_flux3_text_encoder(
            model_config["text_encoder_path"], self.device, self.dtype, model_config.get("tokenizer_path")
        )
        for caption in tqdm(sorted(prompts), desc="Caching prompts", disable=not self.is_main_process):
            self.text_cache[caption] = encoder.encode(caption, self.device, self.dtype).cpu()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        print("Text embeddings pre-cached and Text Encoder unloaded from VRAM.", flush=True)
        logger.info(
            "Pre-cached %d images and %d prompt embeddings (text windows: %s tokens).",
            len(self.samples),
            len(self.text_cache),
            sorted({embeds.shape[0] for embeds in self.text_cache.values()}),
        )

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self) -> None:
        print("=== Step 2/2: Loading FLUX.3 [klein] DiT ===", flush=True)
        model_config = self.config["model"]
        lora_config = self.config.get("lora", {})
        train_config = self.config.get("train", {})
        grad_precision = str(train_config.get("int8_grad_precision", "bf16"))
        self.transformer, load_info = load_flux3_transformer(
            model_config["transformer_path"], self.device, self.dtype, grad_precision=grad_precision
        )
        self.run_metadata["base_load"] = load_info
        self.run_metadata["int8_grad_precision"] = grad_precision
        if bool(train_config.get("gradient_checkpointing", True)):
            self.transformer.enable_gradient_checkpointing()

        self.lora_network = Flux3LoRANetwork(
            self.transformer,
            rank=int(lora_config.get("rank", 16)),
            alpha=float(lora_config.get("alpha", lora_config.get("rank", 16))),
            targets=str(lora_config.get("targets", "blocks")),
            prefix=lora_config.get("prefix", "diffusion_model."),
        )
        self.lora_dtype = getattr(torch, str(lora_config.get("dtype", "float32")))
        self.lora_network.set_lora_dtype(self.device, self.lora_dtype)
        trainable = sum(p.numel() for p in self.lora_network.trainable_parameters())
        self.run_metadata.update(
            {
                "lora_layers": len(self.lora_network.lora_layers),
                "trainable_parameters": trainable,
                "lora_dtype": str(self.lora_dtype).replace("torch.", ""),
            }
        )
        logger.info("LoRA: %.1fM trainable parameters in %s", trainable / 1e6, self.lora_dtype)

    def setup_optimizer(self) -> None:
        train_config = self.config["train"]
        name = str(train_config.get("optimizer", "automagic3")).lower()
        weight_decay = float(train_config.get("weight_decay", 0.0))
        lr = float(train_config.get("learning_rate", 1e-6))

        if name in {"automagic3", "automagic", "automagic_v3"}:
            from maltrainer.optimizers import Automagic3

            # As for Krea 2: attention, MLP and I/O each adapt their own rate, over a 16-step
            # sign window, which suits batch-1 gradients better than 8.
            groups = self.lora_network.param_groups(bool(train_config.get("automagic_param_groups", True)))
            kwargs: dict[str, Any] = {
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
            self.optimizer: Any = Automagic3(groups, **kwargs)
            self.run_metadata["optimizer_resolved"] = {
                "name": "automagic3",
                "groups": [g["name"] for g in groups],
                **kwargs,
            }
        elif name.startswith("prodigy"):
            self.optimizer, kwargs = build_prodigy(
                self.lora_network.trainable_parameters(), train_config, lr, weight_decay
            )
            self.run_metadata["optimizer_resolved"] = {"name": "prodigy", **kwargs}
        elif name in {"adamw8bit", "adam8bit"}:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(
                self.lora_network.trainable_parameters(), lr=lr, weight_decay=weight_decay
            )
            self.run_metadata["optimizer_resolved"] = {"name": name, "lr": lr}
        else:
            self.optimizer = AdamW(self.lora_network.trainable_parameters(), lr=lr, weight_decay=weight_decay)
            self.run_metadata["optimizer_resolved"] = {"name": "adamw", "lr": lr}
        logger.info("Optimizer: %s", self.run_metadata["optimizer_resolved"])

        warmup = int(train_config.get("learning_rate_warmup_steps", 0) or 0)
        if warmup > 0:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lambda step: min((step + 1) / warmup, 1.0)
            )
            logger.info("Learning-rate warmup over %d steps.", warmup)

    # ------------------------------------------------------------------ training

    def _sample_sigma(self) -> torch.Tensor:
        """Flow-matching sigma for one sample: a uniform or logit-normal draw, then the FLUX.3 shift."""
        train_config = self.config["train"]
        distribution = str(train_config.get("timestep_distribution", "uniform")).lower()
        if distribution in {"logit_normal", "logit-normal", "sigmoid"}:
            mean = float(train_config.get("timestep_logit_mean", 0.0))
            std = float(train_config.get("timestep_logit_std", 1.0))
            base = torch.sigmoid(torch.randn(1, device=self.device) * std + mean)
        elif distribution == "uniform":
            base = torch.rand(1, device=self.device)
        else:
            raise ValueError(f"train.timestep_distribution must be uniform or logit_normal, got {distribution!r}")
        shift = float(train_config.get("timestep_shift", FLUX3_SHIFT))
        sigma = shift * base / (1.0 + (shift - 1.0) * base)
        lo = float(train_config.get("min_sigma", 0.001))
        hi = float(train_config.get("max_sigma", 0.999))
        return sigma.clamp(lo, hi)

    def _epoch_order(self) -> list[int]:
        repeats = max(int(self.config["dataset"].get("repeats", 1) or 1), 1)
        order = [i for i in range(len(self.samples)) for _ in range(repeats)]
        random.shuffle(order)
        world = self.accelerator.num_processes
        return order[self.accelerator.process_index :: world] if world > 1 else order

    def _training_step(self, index: int) -> torch.Tensor:
        train_config = self.config["train"]
        key, caption = self.samples[index]
        dropout = float(train_config.get("caption_dropout", 0.0) or 0.0)
        if dropout > 0 and torch.rand(()).item() < dropout:
            # The empty prompt in its own 80-token window, exactly the negative prompt CFG uses.
            caption = ""
        x0 = self.latent_cache[key].to(self.device, torch.float32).unsqueeze(0)
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)

        sigma = self._sample_sigma()
        noise = torch.randn_like(x0)
        s = sigma.view(1, 1, 1, 1)
        noisy = (1.0 - s) * x0 + s * noise
        pred = self.lora_network(noisy.to(self.dtype), sigma, context)
        loss = torch.nn.functional.mse_loss(pred.float(), (noise - x0).float())
        strength = float(train_config.get("loss_weight_strength", 1.0))
        return loss * strength if strength != 1.0 else loss

    def train(self) -> None:
        train_config = self.config["train"]
        save_every = int(train_config.get("save_every", 250) or 0)
        self.lora_network, self.optimizer = self.accelerator.prepare(self.lora_network, self.optimizer)
        network = self.accelerator.unwrap_model(self.lora_network)

        ema_decay = float(train_config.get("ema_decay", 0.0) or 0.0)
        if ema_decay > 0:
            self.ema = ParameterEMA(
                network.trainable_parameters(), decay=ema_decay, warmup=bool(train_config.get("ema_warmup", True))
            )
            self.run_metadata["ema_decay"] = ema_decay
            logger.info("EMA of the LoRA weights: decay %.4f (checkpoints save the average)", ema_decay)

        steps_per_epoch, num_epochs, total_steps = plan_training_steps(len(self._epoch_order()), train_config)
        self.run_metadata.update(
            {
                "steps_per_epoch": steps_per_epoch,
                "planned_total_steps": total_steps,
                "dataset_samples": len(self.samples),
            }
        )
        self.run_metadata["resolved_sigma_shift"] = float(train_config.get("timestep_shift", FLUX3_SHIFT))
        logger.info(
            "Starting FLUX.3 [klein] LoRA training: %d epochs, %d total steps; timesteps %s, shift %.2f",
            num_epochs,
            total_steps,
            train_config.get("timestep_distribution", "uniform"),
            self.run_metadata["resolved_sigma_shift"],
        )

        progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not self.is_main_process)
        global_step = 0
        loss = None
        for epoch in range(num_epochs):
            self.lora_network.train()
            for index in self._epoch_order():
                with self.accelerator.accumulate(self.lora_network):
                    loss = self._training_step(index)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            network.trainable_parameters(), float(train_config.get("max_grad_norm", 1.0))
                        )
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
                progress_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "step": global_step, "epoch": f"{epoch + 1}/{num_epochs}"}
                )
                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        progress_bar.close()
        self.save_checkpoint("final")
        logger.info("FLUX.3 [klein] LoRA training completed!")

    def _log_step(self, step: int, total: int, loss: torch.Tensor, epoch: int, num_epochs: int) -> None:
        group = self.optimizer.param_groups[0]
        inner = getattr(self.optimizer, "optimizer", self.optimizer)
        prefix = (
            f"Step: {step}/{total} [{int(step / total * 100)}%], "
            f"Loss: {loss.item():.6f}, Epoch: {epoch + 1}/{num_epochs}, "
        )
        if hasattr(inner, "get_avg_learning_rate"):
            rates = ", ".join(
                f"{g.get('name', i)} {lr:.2e}"
                for i, (g, lr) in enumerate(zip(inner.param_groups, inner.get_learning_rates(), strict=False))
            )
            self.progress_logger.info("%sLR: %.2e (automagic3: %s)", prefix, inner.get_avg_learning_rate(), rates)
        elif group.get("d") is not None:
            self.progress_logger.info(
                "%sLR: %.2e (mult %.3f x d %.2e)", prefix, group["lr"] * group["d"], group["lr"], group["d"]
            )
        else:
            self.progress_logger.info("%sLR: %.2e", prefix, group["lr"])

    def save_checkpoint(self, step: int | str) -> None:
        if not self.is_main_process:
            return
        output_config = self.config.get("output", {})
        output_dir = Path(output_config["path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        name = output_config.get("name") or output_config.get("lora_name") or "flux3_klein_lora"
        name = name[: -len(".safetensors")] if name.endswith(".safetensors") else name
        save_path = output_dir / (f"{name}.safetensors" if step == "final" else f"{name}_step_{step}.safetensors")
        save_dtype = getattr(torch, str(output_config.get("dtype", "bfloat16")))
        network = self.accelerator.unwrap_model(self.lora_network)
        if self.ema is not None:
            with self.ema.applied():
                state_dict = network.get_state_dict(save_dtype)
        else:
            state_dict = network.get_state_dict(save_dtype)
        runtime = dict(
            self.run_metadata,
            saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            save_dtype=str(save_dtype).replace("torch.", ""),
        )
        metadata = build_metadata(
            self.config, step=step, runtime=runtime, embed_config=bool(output_config.get("embed_config", True))
        )
        save_file(state_dict, str(save_path), metadata=metadata)
        self.last_saved_path = str(save_path)
        logger.info("Saved LoRA checkpoint to %s", save_path)

    # ------------------------------------------------------------------ dry run / entry

    def simulate(self) -> None:
        """Plan from the real dataset folder and tick through the steps without loading models."""
        dataset_config = self.config["dataset"]
        repeats = max(int(dataset_config.get("repeats", 1) or 1), 1)
        samples = max(count_media_files(dataset_config.get("path", ""), ["image"]), 1) * repeats
        steps_per_epoch, num_epochs, total_steps = plan_training_steps(samples, self.config["train"])
        logger.info(
            "Dry run: %d samples, %d steps/epoch -> %d epochs, %d total steps",
            samples,
            steps_per_epoch,
            num_epochs,
            total_steps,
        )
        with tqdm(total=total_steps, desc="Training Steps (dry run)", disable=not self.is_main_process) as bar:
            for step in range(1, total_steps + 1):
                time.sleep(self.dry_run_step_time)
                bar.update(1)
                self.progress_logger.info(
                    "Step: %d/%d [%d%%], Loss: 0.000000, Epoch: %d/%d, LR: 0.00e+00 (dry run)",
                    step,
                    total_steps,
                    int(step / total_steps * 100),
                    math.ceil(step / steps_per_epoch),
                    num_epochs,
                )
        logger.info("Dry run complete: would save final checkpoint at step %d", total_steps)

    def run(self) -> None:
        try:
            set_seed(self.config.get("seed", 42))
            if self.dry_run:
                self.simulate()
                return
            self.precache_dataset()
            if not self.samples:
                raise ValueError("No usable images: every file in the dataset failed to load")
            self.load_transformer_and_lora()
            self.setup_optimizer()
            self.train()
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Exiting...")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error("An error occurred during training: %s", e, exc_info=True)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a FLUX.3 [klein] LoRA in maltrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without loading models")
    parser.add_argument("--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.get("logging", {}))
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])
    Flux3KleinLoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
