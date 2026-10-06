#!/usr/bin/env python3
"""Train a Supra2-IMG LoRA or a full DiT finetune. Images are center-cropped and resized in memory."""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from maltrainer.data.supra_dataset import Supra2ImgDataset
from maltrainer.models.supra2_img.lora import SupraLoRANetwork
from maltrainer.models.supra2_img.model import load_supra2_img
from maltrainer.models.supra2_img.text_encoder import load_flan_t5
from maltrainer.models.supra2_img.vae import encode_latents, load_sd_vae
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.logging_utils import setup_logging
from maltrainer.utils.lora_metadata import build_metadata, git_commit

logger = logging.getLogger(__name__)


def latest_step_checkpoint(output_dir: Path, name: str) -> Optional[Path]:
    """Newest ``{name}_step_N.safetensors`` that has a matching optimizer sidecar."""
    prefix = f"{name}_step_"
    best_step = -1
    best_path: Optional[Path] = None
    if not output_dir.is_dir():
        return None
    for path in output_dir.glob(f"{name}_step_*.safetensors"):
        suffix = path.stem[len(prefix) :]
        if not suffix.isdigit():
            continue
        if not path.with_name(path.stem + ".resume.pt").is_file():
            continue
        step = int(suffix)
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def resume_sidecar(weights: Path, output_dir: Path, name: str) -> Path:
    direct = weights.with_name(weights.stem + ".resume.pt")
    if direct.is_file():
        return direct
    from maltrainer.utils.lora_metadata import read_metadata

    step = str(read_metadata(weights).get("step", ""))
    if step.isdigit():
        candidate = output_dir / f"{name}_step_{step}.resume.pt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No optimizer state for {weights}. A weights-only file cannot continue a run."
    )


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _move_optimizer_state(optimizer, device: torch.device) -> None:
    inner = getattr(optimizer, "optimizer", optimizer)
    for state in inner.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def plan_training_steps(num_batches: int, train_config: dict) -> Tuple[int, int, int]:
    grad_accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)
    steps_per_epoch = max(math.ceil(num_batches / grad_accum), 1)
    max_steps = int(train_config.get("max_steps", 0) or 0)
    if max_steps > 0:
        return steps_per_epoch, math.ceil(max_steps / steps_per_epoch), max_steps
    num_epochs = max(int(train_config.get("num_epochs", 10) or 1), 1)
    return steps_per_epoch, num_epochs, num_epochs * steps_per_epoch


class Supra2ImgLoRATrainer:
    def __init__(self, config: dict, dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run = dry_run
        self.dry_run_step_time = dry_run_step_time
        self.lr_scheduler = None
        self.run_metadata: Dict[str, object] = {
            "model_type": "supra2_img",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        mode = str(config.get("train", {}).get("mode", "lora")).lower()
        if mode not in {"lora", "finetune"}:
            raise ValueError(f"train.mode must be 'lora' or 'finetune', got {mode!r}")
        self.finetune = mode == "finetune"
        self.resume_override: Optional[str] = None
        self.resume_weights: Optional[Path] = None
        self.sidecar: Dict[str, torch.Tensor] = {}
        self.lora_network = None
        self.latent_cache: Dict[str, torch.Tensor] = {}
        self.text_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.accelerator = Accelerator(
            mixed_precision=config.get("mixed_precision", "bf16"),
            gradient_accumulation_steps=config.get("train", {}).get("gradient_accumulation_steps", 1),
        )
        self.is_main_process = self.accelerator.is_main_process
        log_dir = Path(config.get("output", {}).get("path", "./output"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)
        self.progress_logger = logging.getLogger("progress")
        self.progress_logger.setLevel(logging.INFO)
        self.progress_logger.addHandler(file_handler)
        self.progress_logger.propagate = False

    def precache_dataset(self) -> None:
        model_config = self.config["model"]
        dataset_config = self.config["dataset"]
        logger.info("Pre-caching Supra2-IMG latents and Flan-T5 embeddings")
        temp_dataset = Supra2ImgDataset(
            data_root=dataset_config["path"],
            resolution=dataset_config.get("resolution", 256),
            trigger_token=dataset_config.get("trigger"),
            default_caption=dataset_config.get("default_caption", ""),
            repeats=1,
        )
        if len(temp_dataset) == 0:
            raise RuntimeError(f"No images found in {dataset_config['path']}")

        vae = load_sd_vae(
            model_config["vae_path"],
            model_config["vae_config"],
            device=self.device,
            dtype=self.dtype,
        )
        loader = DataLoader(temp_dataset, batch_size=1, shuffle=False)
        for item in tqdm(loader, desc="Caching VAE latents", disable=not self.is_main_process):
            img_path = item["image_path"][0]
            if img_path in self.latent_cache:
                continue
            pixels = item["pixel_values"].to(self.device, dtype=self.dtype)
            self.latent_cache[img_path] = encode_latents(vae, pixels).detach().cpu()
        del vae
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        text_encoder = load_flan_t5(
            model_config["text_encoder_path"],
            model_config["text_encoder_config"],
            model_config["tokenizer_path"],
            device=self.device,
        )
        captions = list(set(temp_dataset.captions))
        if "" not in captions:
            captions.append("")
        sample = self._sample_config()
        if sample is not None:
            for _, prompt in sample["prompts"]:
                if prompt not in captions:
                    captions.append(prompt)
        for caption in tqdm(captions, desc="Caching prompts", disable=not self.is_main_process):
            hidden, mask = text_encoder.encode([caption], device=self.device)
            self.text_cache[caption] = (hidden.detach().cpu(), mask.detach().cpu())
        del text_encoder
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Cached %d latents and %d prompts", len(self.latent_cache), len(self.text_cache))

    def _output_dir(self) -> Path:
        return Path(self.config["output"]["path"])

    def _output_name(self) -> str:
        default_name = "supra2_img_finetune" if self.finetune else "supra2_img_lora"
        name = str(self.config.get("output", {}).get("name") or default_name)
        if name.endswith(".safetensors"):
            name = name[: -len(".safetensors")]
        return name

    def _resume_setting(self) -> str:
        if self.resume_override:
            return str(self.resume_override).strip()
        value = self.config.get("train", {}).get("resume", "")
        return "" if value is None else str(value).strip()

    def resolve_resume_checkpoint(self) -> Optional[Path]:
        setting = self._resume_setting()
        if not setting:
            return None
        output_dir = self._output_dir()
        name = self._output_name()
        if setting.lower() == "latest":
            found = latest_step_checkpoint(output_dir, name)
            if found is None:
                logger.info("No step checkpoint in %s. Starting from step 0.", output_dir)
                return None
            return found
        path = Path(setting)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path

    def load_transformer_and_lora(self) -> None:
        model_config = self.config["model"]
        weight_path = model_config["transformer_path"]
        if self.resume_weights is not None and self.finetune:
            weight_path = self.resume_weights
            logger.info("Loading finetune weights from %s", weight_path)
        transformer, sidecar = load_supra2_img(
            weight_path,
            self.device,
            self.dtype,
            freeze=not self.finetune,
        )
        self.transformer = transformer
        self.sidecar = sidecar
        self.run_metadata["ctx_len"] = int(sidecar.get("ctx_len", 128))
        self.run_metadata["train_mode"] = "finetune" if self.finetune else "lora"
        if self.finetune:
            trainable = sum(param.numel() for param in transformer.parameters())
            self.run_metadata["trainable_parameters"] = trainable
            logger.info("Finetune: %.1fM trainable DiT parameters (VAE and text encoder stay frozen)", trainable / 1e6)
            return

        lora_config = self.config.get("lora", {})
        self.lora_network = SupraLoRANetwork(
            transformer=transformer,
            rank=int(lora_config.get("rank", 16)),
            alpha=float(lora_config.get("alpha", 16.0)),
            target_modules=lora_config.get("target_modules"),
            prefix=str(lora_config.get("prefix", "")),
        )
        self.lora_dtype = getattr(torch, str(lora_config.get("dtype", "float32")))
        self.lora_network.set_lora_dtype(self.device, self.lora_dtype)
        if self.resume_weights is not None:
            self.lora_network.load_weights(self.resume_weights)
        trainable = sum(param.numel() for param in self.lora_network.trainable_parameters())
        self.run_metadata["trainable_parameters"] = trainable
        self.run_metadata["lora_layers"] = len(self.lora_network.lora_layers)
        logger.info("LoRA adapters: %.2fM trainable parameters", trainable / 1e6)

    def create_dataset(self) -> None:
        dataset_config = self.config["dataset"]
        self.train_dataset = Supra2ImgDataset(
            data_root=dataset_config["path"],
            resolution=dataset_config.get("resolution", 256),
            trigger_token=dataset_config.get("trigger"),
            default_caption=dataset_config.get("default_caption", ""),
            repeats=dataset_config.get("repeats", 1),
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config["train"].get("batch_size", 1),
            shuffle=True,
            num_workers=dataset_config.get("num_workers", 0),
        )

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        train_config = self.config.get("train", {})
        distribution = str(train_config.get("timestep_distribution", "uniform")).lower()
        if distribution in {"sigmoid", "logit_normal"}:
            scale = float(train_config.get("sigmoid_scale", 1.0))
            timesteps = torch.sigmoid(torch.randn(batch_size, device=self.device, dtype=torch.float32) * scale)
        else:
            timesteps = torch.rand(batch_size, device=self.device, dtype=torch.float32)
        return timesteps.clamp(0.001, 0.999)

    def setup_optimizer(self) -> None:
        train_config = self.config["train"]
        lr = float(train_config.get("learning_rate", 1e-4))
        optimizer_name = str(train_config.get("optimizer", "adamw8bit")).lower()
        weight_decay = float(train_config.get("weight_decay", 0.01))
        params = (
            [param for param in self.transformer.parameters() if param.requires_grad]
            if self.finetune
            else list(self.lora_network.trainable_parameters())
        )

        if optimizer_name in {"adamw8bit", "adam8bit"}:
            try:
                import bitsandbytes as bnb

                optimizer_cls = bnb.optim.AdamW8bit if optimizer_name == "adamw8bit" else bnb.optim.Adam8bit
                self.optimizer = optimizer_cls(params, lr=lr, weight_decay=weight_decay)
                self.run_metadata["optimizer_resolved"] = {"name": optimizer_name, "lr": lr}
            except Exception as exc:
                logger.warning("Bitsandbytes not available (%s). Falling back to AdamW.", exc)
                self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
                self.run_metadata["optimizer_resolved"] = {"name": "adamw", "lr": lr}
        elif optimizer_name.startswith("prodigy"):
            try:
                from prodigyopt import Prodigy

                self.optimizer = Prodigy(
                    params,
                    lr=float(train_config.get("prodigy_lr", 1.0)),
                    weight_decay=weight_decay,
                    d_coef=float(train_config.get("prodigy_d_coef", 1.0)),
                )
                self.run_metadata["optimizer_resolved"] = {"name": "prodigy"}
            except Exception as exc:
                logger.warning("Prodigy not available (%s). Falling back to AdamW.", exc)
                self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
                self.run_metadata["optimizer_resolved"] = {"name": "adamw", "lr": lr}
        elif optimizer_name in {"automagic3", "automagic", "automagic_v3"}:
            from maltrainer.optimizers import Automagic3

            automagic_params = {
                "lr": float(train_config.get("learning_rate", 1e-6)),
                "min_lr": float(train_config.get("automagic_min_lr", 1e-8)),
                "max_lr": float(train_config.get("automagic_max_lr", 1e3)),
                "beta2": float(train_config.get("automagic_beta2", 0.999)),
                "weight_decay": weight_decay,
                "polarity_history": int(train_config.get("automagic_polarity_history", 8)),
            }
            self.optimizer = Automagic3(params, **automagic_params)
            self.run_metadata["optimizer_resolved"] = {"name": "automagic3", **automagic_params}
        else:
            self.optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
            self.run_metadata["optimizer_resolved"] = {"name": "adamw", "lr": lr}

    def _restore_training_state(self, blob_path: Path) -> int:
        blob = torch.load(blob_path, map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(blob["optimizer"])
        _move_optimizer_state(self.optimizer, self.device)
        if "torch_rng" in blob:
            torch.set_rng_state(blob["torch_rng"])
        cuda_rng = blob.get("cuda_rng")
        if cuda_rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)
        step = int(blob["global_step"])
        logger.info("Resuming from step %d (%s)", step, blob_path)
        return step

    def train(self) -> None:
        train_config = self.config["train"]
        save_every = int(train_config.get("save_every", 250) or 0)
        keep_last = int(train_config.get("keep_last", 3) or 0)
        caption_dropout = float(train_config.get("caption_dropout", 0.0))
        module = self.transformer if self.finetune else self.lora_network
        module, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            module, self.optimizer, self.train_dataloader
        )
        if self.finetune:
            self.transformer = module
        else:
            self.lora_network = module
        global_step = 0
        if self.resume_weights is not None:
            blob_path = resume_sidecar(self.resume_weights, self._output_dir(), self._output_name())
            global_step = self._restore_training_state(blob_path)
        steps_per_epoch, num_epochs, total_steps = plan_training_steps(len(self.train_dataloader), train_config)
        self.run_metadata["planned_total_steps"] = total_steps
        self.run_metadata["dataset_samples"] = len(self.train_dataset)
        if global_step >= total_steps:
            logger.info("Checkpoint step %d has already reached the planned %d steps.", global_step, total_steps)
            return
        if keep_last > 0:
            logger.info("Rolling checkpoints: keeping the last %d", keep_last)
        if global_step == 0:
            self.save_previews(0)
        progress_bar = tqdm(
            total=total_steps,
            initial=global_step,
            desc="Training Steps",
            disable=not self.is_main_process,
        )

        try:
            for epoch in range(num_epochs):
                if global_step >= total_steps:
                    break
                if steps_per_epoch > 0 and (epoch + 1) * steps_per_epoch <= global_step:
                    continue
                if self.finetune:
                    self.transformer.train()
                else:
                    self.lora_network.train()
                    self.transformer.eval()
                for batch in self.train_dataloader:
                    with self.accelerator.accumulate(self.transformer if self.finetune else self.lora_network):
                        latents = torch.cat([self.latent_cache[path] for path in batch["image_path"]], dim=0)
                        latents = latents.to(device=self.device, dtype=self.dtype)
                        captions = []
                        for caption in batch["caption"]:
                            if caption_dropout > 0 and torch.rand(1).item() < caption_dropout:
                                captions.append("")
                            else:
                                captions.append(caption)
                        hidden = torch.cat([self.text_cache[caption][0] for caption in captions], dim=0)
                        mask = torch.cat([self.text_cache[caption][1] for caption in captions], dim=0)
                        hidden = hidden.to(device=self.device, dtype=self.dtype)
                        mask = mask.to(device=self.device, dtype=self.dtype)

                        timesteps = self._sample_timesteps(latents.shape[0])
                        noise = torch.randn_like(latents)
                        weights = timesteps.view(-1, 1, 1, 1).to(dtype=latents.dtype)
                        noisy = (1.0 - weights) * noise + weights * latents
                        target = latents - noise
                        use_amp = self.device.type == "cuda"
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                            predicted = self.transformer(noisy, timesteps.to(dtype=latents.dtype), hidden, mask)
                        loss = F.mse_loss(predicted.float(), target.float())
                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.transformer.parameters() if self.finetune else self.lora_network.trainable_parameters(),
                                float(train_config.get("max_grad_norm", 1.0)),
                            )
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        global_step += 1
                        progress_bar.update(1)
                        current_lr = self.optimizer.param_groups[0]["lr"]
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
                        if save_every > 0 and global_step % save_every == 0 and global_step < total_steps:
                            self.save_checkpoint(global_step)
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})
                    if global_step >= total_steps:
                        break
        except KeyboardInterrupt:
            if global_step > 0:
                logger.info("Interrupted at step %d. Saving a resumable checkpoint.", global_step)
                self.save_checkpoint(global_step)
            logger.info("Continue this run with --resume latest")
            raise
        finally:
            progress_bar.close()

        self.save_checkpoint(global_step, final=True)
        logger.info("Supra2-IMG %s training completed", "finetune" if self.finetune else "LoRA")

    def _weights_state_dict(self, save_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        if self.finetune:
            model = self.accelerator.unwrap_model(self.transformer)
            state_dict = {key: value.detach().cpu().to(save_dtype) for key, value in model.state_dict().items()}
            if "uncond_text" in self.sidecar:
                state_dict["__uncond_text__"] = self.sidecar["uncond_text"].detach().cpu()
            if "uncond_mask" in self.sidecar:
                state_dict["__uncond_mask__"] = self.sidecar["uncond_mask"].detach().cpu()
            return state_dict
        network = self.accelerator.unwrap_model(self.lora_network)
        return network.get_state_dict(save_dtype=save_dtype)

    def _prune_step_checkpoints(self, output_dir: Path, name: str, keep_last: int) -> None:
        if keep_last <= 0:
            return
        prefix = f"{name}_step_"
        numbered = []
        for path in output_dir.glob(f"{name}_step_*.safetensors"):
            suffix = path.stem[len(prefix) :]
            if suffix.isdigit():
                numbered.append((int(suffix), path))
        numbered.sort()
        for step, path in numbered[:-keep_last]:
            path.unlink(missing_ok=True)
            path.with_name(path.stem + ".resume.pt").unlink(missing_ok=True)
            logger.info("Removed checkpoint step %d (keeping the last %d)", step, keep_last)

    def save_checkpoint(self, step: int, *, final: bool = False) -> None:
        if not self.is_main_process:
            return
        output_dir = self._output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        name = self._output_name()
        step_path = output_dir / f"{name}_step_{step}.safetensors"
        default_dtype = "float32" if self.finetune else "bfloat16"
        save_dtype = getattr(torch, str(self.config.get("output", {}).get("dtype", default_dtype)))
        state_dict = self._weights_state_dict(save_dtype)
        runtime = dict(self.run_metadata)
        runtime["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        runtime["save_dtype"] = str(save_dtype).replace("torch.", "")
        runtime["global_step"] = step
        metadata = build_metadata(
            self.config,
            step=step,
            runtime=runtime,
            embed_config=bool(self.config.get("output", {}).get("embed_config", True)),
        )
        if self.finetune:
            metadata["patch"] = "2"
            metadata["ctx_len"] = str(self.run_metadata.get("ctx_len", 128))
        weights_partial = step_path.with_name(step_path.name + ".partial")
        save_file(state_dict, str(weights_partial), metadata=metadata)
        os.replace(weights_partial, step_path)
        resume_path = step_path.with_name(step_path.stem + ".resume.pt")
        resume_partial = resume_path.with_name(resume_path.name + ".partial")
        torch.save(
            {
                "global_step": int(step),
                "optimizer": _cpu_tree(self.optimizer.state_dict()),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            resume_partial,
        )
        os.replace(resume_partial, resume_path)
        if final:
            final_path = output_dir / f"{name}.safetensors"
            if final_path.exists():
                final_path.unlink()
            try:
                os.link(step_path, final_path)
            except OSError:
                shutil.copy2(step_path, final_path)
            logger.info("Published %s", final_path)
        keep_last = int(self.config.get("train", {}).get("keep_last", 3) or 0)
        self._prune_step_checkpoints(output_dir, name, keep_last)
        logger.info("Saved %s checkpoint step %d to %s", "finetune" if self.finetune else "LoRA", step, step_path)
        self.save_previews(step)

    def _sample_config(self) -> Optional[dict]:
        sample = self.config.get("sample", {})
        if sample is False or sample is None:
            return None
        if not isinstance(sample, dict):
            sample = {}
        if sample.get("enabled", True) is False:
            return None
        return {
            "seed": int(sample.get("seed", 0)),
            "cfg": float(sample.get("cfg", 3.0)),
            "steps": max(int(sample.get("steps", 50) or 1), 1),
            "resolution": int(self.config.get("dataset", {}).get("resolution", 256)),
            "prompts": [
                ("man", str(sample.get("man", "a portrait photo of a man's face"))),
                ("woman", str(sample.get("woman", "a portrait photo of a woman's face"))),
            ],
        }

    def _batched_uncond(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if "uncond_text" in self.sidecar and "uncond_mask" in self.sidecar:
            text = self.sidecar["uncond_text"]
            mask = self.sidecar["uncond_mask"]
        else:
            text, mask = self.text_cache[""]
        if text.dim() == 2:
            text = text.unsqueeze(0)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        return text, mask

    @torch.no_grad()
    def _flow_sample(self, prompt: str, sample: dict) -> torch.Tensor:
        from maltrainer.models.supra2_img.model import LATENT_CH

        hidden, mask = self.text_cache[prompt]
        hidden = hidden[:1].to(device=self.device, dtype=torch.float32)
        mask = mask[:1].to(device=self.device, dtype=torch.float32)
        uncond, uncond_mask = self._batched_uncond()
        uncond = uncond[:1].to(device=self.device, dtype=torch.float32)
        uncond_mask = uncond_mask[:1].to(device=self.device, dtype=torch.float32)
        ctx = torch.cat([hidden, uncond], 0)
        ctx_mask = torch.cat([mask, uncond_mask], 0)
        latent_size = sample["resolution"] // 8
        generator = torch.Generator(device=self.device)
        generator.manual_seed(sample["seed"])
        latent = torch.randn(1, LATENT_CH, latent_size, latent_size, device=self.device, generator=generator)
        steps = sample["steps"]
        cfg = sample["cfg"]
        dt = 1.0 / steps
        use_amp = self.device.type == "cuda"
        for index in range(steps):
            timestep = torch.full((1,), index * dt, device=self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                both = self.transformer(
                    torch.cat([latent, latent], 0),
                    torch.cat([timestep, timestep], 0),
                    ctx,
                    ctx_mask,
                )
            cond, uncond_v = both.float().chunk(2, 0)
            velocity = uncond_v + cfg * (cond - uncond_v)
            latent = latent + dt * velocity
        return latent

    @torch.no_grad()
    def save_previews(self, step: int) -> None:
        sample = self._sample_config()
        if sample is None or not self.is_main_process:
            return
        from PIL import Image

        from maltrainer.models.supra2_img.model import VAE_SCALE
        from maltrainer.models.supra2_img.vae import load_sd_vae

        was_training = self.transformer.training
        was_lora = self.lora_network.training if self.lora_network is not None else False
        self.transformer.eval()
        if self.lora_network is not None:
            self.lora_network.eval()
        try:
            latents = [(label, self._flow_sample(prompt, sample)) for label, prompt in sample["prompts"]]
            model_config = self.config["model"]
            vae = load_sd_vae(
                model_config["vae_path"],
                model_config["vae_config"],
                device=self.device,
                dtype=self.dtype,
            )
            preview_dir = self._output_dir() / "samples"
            preview_dir.mkdir(parents=True, exist_ok=True)
            use_amp = self.device.type == "cuda"
            for label, latent in latents:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    decoded = vae.decode(latent.to(dtype=self.dtype) / VAE_SCALE).sample
                pixels = ((decoded[0].float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
                path = preview_dir / f"step_{step:06d}_{label}.png"
                Image.fromarray(pixels).save(path)
                logger.info("Saved %s preview step %d to %s", label, step, path)
            del vae
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("Preview sampling failed at step %d", step)
        finally:
            self.transformer.train(was_training)
            if self.lora_network is not None:
                self.lora_network.train(was_lora)

    def simulate(self) -> None:
        self.create_dataset()
        train_config = self.config["train"]
        steps_per_epoch, num_epochs, total_steps = plan_training_steps(len(self.train_dataloader), train_config)
        logger.info(
            "Dry run: %d samples, %d steps/epoch, %d total steps",
            len(self.train_dataset),
            steps_per_epoch,
            total_steps,
        )

    def run(self) -> None:
        set_seed(self.config.get("seed", 42))
        if self.dry_run:
            self.simulate()
            return
        self.resume_weights = self.resolve_resume_checkpoint()
        self.create_dataset()
        self.precache_dataset()
        self.load_transformer_and_lora()
        self.setup_optimizer()
        self.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Supra2-IMG LoRA or finetune in Maltrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Count the dataset and stop before loading models")
    parser.add_argument("--dry-run-step-time", type=float, default=0.025, help="Unused; kept so the CLI matches the other trainers")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint .safetensors to continue, or 'latest'")
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}))
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])
    trainer = Supra2ImgLoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time)
    trainer.resume_override = args.resume or None
    trainer.run()


if __name__ == "__main__":
    main()
