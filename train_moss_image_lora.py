#!/usr/bin/env python3
"""
MOSS-Image LoRA training for maltrainer.

Trains MOSS-Image Base (OpenMOSS, 5B) on ComfyUI's single files (Comfy-Org/MOSS-Image): the DiT
(moss_image_base_bf16, or _int8_convrot), the Flux2 VAE and the Qwen3.5-9B text encoder
(qwen3.5_9b_int8_convrot). Latents and caption embeddings are cached once, then the encoders
are unloaded before the DiT loads.

Conventions follow ComfyUI's MOSS-Image (comfy/ldm/moss_image, ModelType.FLOW) and the OpenMOSS
technical report (appendix A.2): the DiT gets sigma in [0, 1] (embedded as 1000 * sigma) and
predicts v = noise - x0, under a plain per-token MSE. By default sigmas follow the report's
final pretraining interval: logit-normal (mean 1.09, std 0.95 on logit sigma), cut at 0.95 and
rescaled onto [0, 1], 1 % uniform, no shift.

    python train_moss_image_lora.py --config templates/moss_image_automagic_template.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import logging
import math
import random
import threading
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from safetensors.torch import save_file
from torch.optim import AdamW
from tqdm import tqdm

from maltrainer.data.disk_cache import FolderCache, cache_key, content_hash, model_identity
from maltrainer.data.flux3_dataset import load_flux3_image
from maltrainer.data.minimax_dataset import count_media_files, scan_media_sources
from maltrainer.models.flux3.vae import load_flux2_vae_decoder, load_flux2_vae_encoder
from maltrainer.models.moss_image.loading import load_moss_image_transformer
from maltrainer.models.moss_image.lora import DEFAULT_SKIP_BLOCKS, MOSSImageLoRANetwork
from maltrainer.models.moss_image.sampling import sample as moss_sample
from maltrainer.models.moss_image.text_encoder import MAX_SEQUENCE_LENGTH, TEMPLATE, load_moss_image_text_encoder
from maltrainer.utils.compile import compile_blocks
from maltrainer.utils.config import load_config, save_config_to_output
from maltrainer.utils.console_keys import read_key
from maltrainer.utils.ema import ParameterEMA
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.logging_utils import setup_logging
from maltrainer.utils.lora_metadata import build_metadata, git_commit
from maltrainer.utils.preview_menu import run_menu
from maltrainer.utils.previews import Previews
from maltrainer.utils.step_planner import DEFAULT_EPOCHS, Group, folder_counts
from maltrainer.utils.step_planner import report as step_report
from maltrainer.utils.train_monitor import TrainMonitor

logger = logging.getLogger(__name__)

# The technical report's last pretraining interval (Table 6, steps 274,001-310,000): sigma = 1 -
# sigmoid(u), u ~ N(-1.09, 0.95), i.e. logit(sigma) ~ N(1.09, 0.95); truncated at sigma <= 0.95,
# rescaled onto the 1000-step grid {0.001 .. 1}, mixed with 1 % discrete uniform; shift 1.
MOSS_LOGIT_MEAN = 1.09
MOSS_LOGIT_STD = 0.95
MOSS_CUTOFF = 0.95
MOSS_UNIFORM_MIX = 0.01
MOSS_GRID = 1000


def plan_training_steps(num_samples: int, train_config: dict) -> tuple[int, int, int]:
    """(steps_per_epoch, num_epochs, total_steps); ``train.max_steps`` > 0 wins over epochs."""
    grad_accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)
    steps_per_epoch = max(math.ceil(num_samples / grad_accum), 1)
    max_steps = int(train_config.get("max_steps", 0) or 0)
    if max_steps > 0:
        return steps_per_epoch, math.ceil(max_steps / steps_per_epoch), max_steps
    num_epochs = max(int(train_config.get("num_epochs", 10) or 1), 1)
    return steps_per_epoch, num_epochs, num_epochs * steps_per_epoch


class MOSSImageLoRATrainer:
    def __init__(self, config: dict, dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run = dry_run
        self.dry_run_step_time = dry_run_step_time
        self.last_saved_path: str | None = None
        self.lr_scheduler = None
        self.ema: ParameterEMA | None = None
        self.run_metadata: dict[str, object] = {
            "model_type": "moss_image",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "torch_version": torch.__version__,
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.samples: list[tuple[str, str]] = []  # (key, caption)
        self.latent_cache: dict[str, torch.Tensor] = {}
        self.text_cache: dict[str, torch.Tensor] = {}
        # sample: previews of fixed prompts every N steps, rendered with the weights in training
        self.previews = Previews(config.get("sample"), Path(config.get("output", {}).get("path", "./output")))
        self._preview_decoder = None
        # The config file, watched for edits to its sample: section while training runs (set by main)
        self.config_path: Path | None = None
        self._config_mtime: float | None = None
        self._cpu_encoder = None
        self._encoding: threading.Thread | None = None

        train_config = config.get("train", {})
        if int(train_config.get("batch_size", 1) or 1) != 1:
            raise ValueError(
                "MOSS-Image trains at batch_size 1 (every caption is encoded at its own length, and "
                "the RoPE follows it, as at inference); use train.gradient_accumulation_steps for a "
                "larger effective batch."
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
        # dataset.cache: each image folder keeps a .maltrainer_cache of its own encodings.
        use_disk = bool(dataset_config.get("cache", True))

        # Latents: from the disk cache where possible; the VAE loads only if something is new.
        vae_id = model_identity(model_config["vae_path"])
        latents: dict[str, torch.Tensor] = {}
        pending = []
        for source in sources:
            folder = FolderCache(source.path.parent)
            key = (
                cache_key("moss_image_latent/v1", content_hash(source.path), resolution, aspect_mode, vae_id)
                if use_disk
                else None
            )
            latent = folder.get("latents", key) if use_disk else None
            if latent is None:
                pending.append((source, folder, key))
            else:
                latents[str(source.path)] = latent
        if pending:
            vae = load_flux2_vae_encoder(model_config["vae_path"], self.device)
            for source, folder, key in tqdm(pending, desc="Caching latents", disable=not self.is_main_process):
                try:
                    pixels = load_flux3_image(source.path, resolution, aspect_mode).unsqueeze(0).to(self.device)
                    latent = vae.encode(pixels)[0].cpu()
                    latents[str(source.path)] = latent
                    if use_disk:
                        folder.put("latents", key, latent)
                except (OSError, ValueError, RuntimeError) as e:
                    logger.warning("Skipping %s: %s", source.path.name, e)
            del vae
            gc.collect()
            torch.cuda.empty_cache()
        # Path order whatever came from disk, so a resumed run shuffles the same samples.
        sizes: dict[tuple[int, int], int] = {}
        for source in sources:
            latent = latents.get(str(source.path))
            if latent is None:
                continue
            self.latent_cache[str(source.path)] = latent
            self.samples.append((str(source.path), source.caption))
            size = (latent.shape[-1] * 16, latent.shape[-2] * 16)
            sizes[size] = sizes.get(size, 0) + 1
        logger.info(
            "Training sizes (w x h: images): %s", ", ".join(f"{w}x{h}: {n}" for (w, h), n in sorted(sizes.items()))
        )
        logger.info("Latents: %d from the disk cache, %d encoded", len(sources) - len(pending), len(pending))
        print("VAE latents ready.", flush=True)

        # A caption's embedding lives in every folder whose images use it; the empty dropout
        # prompt in the dataset folder itself.
        prompt_folders: dict[str, set] = {}
        for key, caption in self.samples:
            prompt_folders.setdefault(caption, set()).add(Path(key).parent)
        if float(train_config.get("caption_dropout", 0.0) or 0.0) > 0:
            prompt_folders.setdefault("", set()).add(Path(dataset_config["path"]))
        for caption in self._extra_prompts():
            prompt_folders.setdefault(caption, set()).add(Path(dataset_config["path"]))
        prompts = sorted(prompt_folders)
        text_id = cache_key(
            "moss_image_text/v1",
            model_identity(model_config["text_encoder_path"]),
            model_config.get("tokenizer_path"),
            TEMPLATE,
            MAX_SEQUENCE_LENGTH,
            self.dtype,
        )
        pending_prompts = []
        for caption in prompts:
            key = cache_key(text_id, caption)
            folders = [FolderCache(folder) for folder in sorted(prompt_folders[caption])]
            embeds = None
            if use_disk:
                for folder in folders:
                    if folder.has("text", key) and (embeds := folder.get("text", key)) is not None:
                        break
            if embeds is None:
                pending_prompts.append((caption, key, folders))
                continue
            self.text_cache[caption] = embeds
            for folder in folders:  # e.g. a caption now shared with a new folder
                if not folder.has("text", key):
                    folder.put("text", key, embeds)
        if pending_prompts:
            encoder = load_moss_image_text_encoder(
                model_config["text_encoder_path"], self.device, self.dtype, model_config.get("tokenizer_path")
            )
            for caption, key, folders in tqdm(
                pending_prompts, desc="Caching prompts", disable=not self.is_main_process
            ):
                self.text_cache[caption] = encoder.encode(caption, self.device, self.dtype).cpu()
                if use_disk:
                    for folder in folders:
                        folder.put("text", key, self.text_cache[caption])
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
        logger.info(
            "Prompt embeddings: %d from the disk cache, %d encoded",
            len(prompts) - len(pending_prompts),
            len(pending_prompts),
        )
        lengths = sorted({embeds.shape[0] for embeds in self.text_cache.values()})
        print("Text embeddings ready.", flush=True)
        logger.info(
            "Pre-cached %d images and %d prompt embeddings (text lengths: %s tokens).",
            len(self.samples),
            len(self.text_cache),
            lengths,
        )

    def _extra_prompts(self) -> list[str]:
        """Captions to encode beside the samples' own: the preview prompts (and the finetune's
        prior-preservation ones)."""
        return self.previews.captions

    # ------------------------------------------------------------------ previews

    @torch.no_grad()
    def _render_preview(self, prompt: str, seed: int) -> Image.Image:
        """One preview with the weights as they are now: the reference sampler, then the VAE decoder."""
        p = self.previews
        if self._preview_decoder is None:
            self._preview_decoder = load_flux2_vae_decoder(self.config["model"]["vae_path"], self.device)
        context = self.text_cache[prompt].to(self.device, self.dtype)
        negative = self.text_cache[p.negative].to(self.device, self.dtype) if p.cfg != 1.0 else None
        with tqdm(total=p.steps, desc=f"Sampling ({p.current})", leave=False, disable=not self.is_main_process) as bar:
            latents = moss_sample(
                self.lora_network,
                context,
                p.height,
                p.width,
                steps=p.steps,
                seed=seed,
                cfg=p.cfg,
                negative=negative,
                device=self.device,
                on_step=lambda done, total: (setattr(bar, "total", total), bar.update(done - bar.n)),
            )
        image = self._preview_decoder.decode(latents)[0]
        return Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"))

    # ------------------------------------------------------------------ live sample: reload

    def _watch_config(self) -> None:
        """At a step boundary: if the config file was saved since the last look, apply its sample:
        section (interval, prompts, settings) without stopping. Other sections are only read at
        the start; edits to them are reported and ignored until the next start."""
        if self.config_path is None or not self.is_main_process:
            return
        try:
            mtime = self.config_path.stat().st_mtime
        except OSError:
            return
        if self._config_mtime is None:
            self._config_mtime = mtime
            return
        if mtime == self._config_mtime:
            return
        self._config_mtime = mtime
        try:
            fresh = load_config(str(self.config_path))
        except (OSError, ValueError) as error:  # e.g. a half-edited YAML file
            logger.warning(
                "[sample] %s could not be read (%s); the previous settings stay", self.config_path.name, error
            )
            return
        ignored = sorted(
            k for k in set(fresh) | set(self.config) if k != "sample" and fresh.get(k) != self.config.get(k)
        )
        if ignored:
            logger.info(
                "[sample] %s changed outside sample: (%s): those apply at the next start",
                self.config_path.name,
                ", ".join(ignored),
            )
        changes = self.previews.update(fresh.get("sample"))
        self.config["sample"] = fresh.get("sample")
        if changes:
            logger.info("[sample] reloaded from %s:\n  %s", self.config_path.name, "\n  ".join(changes))
        self._encode_new_prompts()

    def _encode_new_prompts(self) -> None:
        """Preview prompts not encoded yet, on the CPU in the background (~6 s each): the GPU is
        busy training, and the text encoder (~10 GB) does not fit beside it. The int8 layers run
        their float fallback on the CPU, so these embeddings differ ~2.6% from the GPU path's;
        they serve previews only and never enter the caption cache."""
        missing = [c for c in dict.fromkeys(self.previews.captions) if c not in self.text_cache]
        if not missing or (self._encoding is not None and self._encoding.is_alive()):
            return

        def work():
            try:
                if self._cpu_encoder is None:
                    model_config = self.config["model"]
                    self._cpu_encoder = load_moss_image_text_encoder(
                        model_config["text_encoder_path"],
                        torch.device("cpu"),
                        self.dtype,
                        model_config.get("tokenizer_path"),
                    )
                for caption in missing:
                    started = time.time()
                    self.text_cache[caption] = self._cpu_encoder.encode(caption, torch.device("cpu"), self.dtype).cpu()
                    logger.info(
                        "[sample] encoded on the CPU in %.0fs: %s", time.time() - started, caption or "(empty negative)"
                    )
            except Exception:  # a background thread: log it with its traceback, do not die silently
                logger.exception("[sample] encoding new prompts failed")

        logger.info("[sample] encoding %d new prompt(s) on the CPU in the background", len(missing))
        self._encoding = threading.Thread(target=work, daemon=True)
        self._encoding.start()

    def _preview_menu(self) -> None:
        """Key e: the preview settings menu. Rows to choose from are the labelled variants (a
        finetune with EMA: raw and ema); prompts added mid-run are encoded first if needed."""
        labels = [label for label, _ in self._preview_variants() if label]
        run_menu(self.previews, labels, read_key, ready=self._preview_ready)
        self._encode_new_prompts()

    def _preview_ready(self, prompt: str) -> bool:
        p = self.previews
        return prompt in self.text_cache and (p.cfg == 1.0 or p.negative in self.text_cache)

    def _preview_variants(self):
        """(label, weights context) per row of the preview strip. LoRA checkpoints save the EMA
        average, so the previews show it."""
        if self.ema is not None and hasattr(self.ema, "applied"):
            return [("", self.ema.applied)]
        return [("", contextlib.nullcontext)]

    def _maybe_preview(self, step: int, total: int) -> float:
        """Render the previews if they are due; returns the seconds spent (0 if not)."""
        if not self.is_main_process or not self.previews.due(step, total):
            return 0.0
        was_training = self.lora_network.training
        self.lora_network.eval()
        # hand training's cached-but-unused memory back first, so the render does not stack on it
        gc.collect()
        torch.cuda.empty_cache()
        try:
            return self.previews.render(step, self._render_preview, self._preview_variants(), ready=self._preview_ready)
        finally:
            self.lora_network.train(was_training)
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self):
        print("=== Step 2/2: Loading MOSS-Image DiT ===", flush=True)
        model_config = self.config["model"]
        lora_config = self.config.get("lora", {})
        train_config = self.config.get("train", {})
        grad_precision = str(train_config.get("int8_grad_precision", "bf16"))
        self.transformer, load_info = load_moss_image_transformer(
            model_config["transformer_path"], self.device, self.dtype, grad_precision=grad_precision
        )
        self.run_metadata["base_load"] = load_info
        self.run_metadata["int8_grad_precision"] = grad_precision
        if bool(train_config.get("gradient_checkpointing", True)):
            self.transformer.enable_gradient_checkpointing()

        self.lora_network = MOSSImageLoRANetwork(
            self.transformer,
            rank=int(lora_config.get("rank", 16)),
            alpha=float(lora_config.get("alpha", lora_config.get("rank", 16))),
            targets=str(lora_config.get("targets", "blocks")),
            prefix=lora_config.get("prefix", "diffusion_model."),
            skip_blocks=lora_config.get("skip_blocks", DEFAULT_SKIP_BLOCKS) or (),
        )
        self.lora_dtype = getattr(torch, str(lora_config.get("dtype", "float32")))
        self.lora_network.set_lora_dtype(self.device, self.lora_dtype)
        trainable = sum(p.numel() for p in self.lora_network.trainable_parameters())
        self.run_metadata.update(
            {
                "lora_layers": len(self.lora_network.lora_layers),
                "trainable_parameters": trainable,
                "lora_dtype": str(self.lora_dtype).replace("torch.", ""),
                "lora_skip_blocks": list(self.lora_network.skip_blocks),
            }
        )
        logger.info("LoRA: %.1fM trainable parameters in %s", trainable / 1e6, self.lora_dtype)
        self._maybe_compile()

    def _maybe_compile(self):
        """train.compile: torch.compile each of the 42 blocks (after the LoRA wraps their Linears)."""
        train_config = self.config.get("train", {})
        if not bool(train_config.get("compile", False)):
            self.run_metadata["compiled"] = False
            return
        self.run_metadata["compiled"] = compile_blocks(
            self.transformer.layers, self.device, mode=train_config.get("compile_mode") or None
        )

    def setup_optimizer(self):
        train_config = self.config["train"]
        name = str(train_config.get("optimizer", "automagic3")).lower()
        weight_decay = float(train_config.get("weight_decay", 0.0))
        lr = float(train_config.get("learning_rate", 1e-6))

        if name in {"automagic3", "automagic", "automagic_v3"}:
            from maltrainer.optimizers import Automagic3

            # Attention, MLP and I/O each adapt their own rate. The sign window is 8 steps: on
            # FLUX.3 a 16-step window left the rate stuck near 3e-5 with batch-1 gradients.
            groups = self.lora_network.param_groups(bool(train_config.get("automagic_param_groups", True)))
            kwargs = {
                "lr": lr,
                "min_lr": float(train_config.get("automagic_min_lr", 1e-8)),
                "max_lr": float(train_config.get("automagic_max_lr", 1e3)),
                "beta2": float(train_config.get("automagic_beta2", 0.999)),
                "eps": float(train_config.get("automagic_eps", 1e-30)),
                "clip_threshold": float(train_config.get("automagic_clip_threshold", 1.0)),
                "weight_decay": weight_decay,
                "polarity_history": int(train_config.get("automagic_polarity_history", 8)),
                "fused": bool(train_config.get("automagic_fused", False)),
            }
            self.optimizer = Automagic3(groups, **kwargs)
            self.run_metadata["optimizer_resolved"] = {
                "name": "automagic3",
                "groups": [g["name"] for g in groups],
                **kwargs,
            }
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
            self.optimizer = Prodigy(self.lora_network.trainable_parameters(), **kwargs)
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
        """Flow-matching sigma for one sample (the defaults are the report's final interval).

        1. logit_normal: logit(sigma) ~ N(mean, std), truncated to sigma <= cutoff by inverse-CDF
           sampling and divided by the cutoff; or uniform
        2. with probability ``timestep_uniform_mix``, a uniform draw instead
        3. the shift sigma' = s * sigma / (1 + (s - 1) * sigma); 1 = none
        4. rounded up onto the training grid {1/N .. 1}, N = ``timestep_grid`` (0 = continuous)
        """
        train_config = self.config["train"]
        distribution = str(train_config.get("timestep_distribution", "logit_normal")).lower()
        if distribution in {"logit_normal", "logit-normal", "sigmoid"}:
            mean = float(train_config.get("timestep_logit_mean", MOSS_LOGIT_MEAN))
            std = float(train_config.get("timestep_logit_std", MOSS_LOGIT_STD))
            cutoff = float(train_config.get("timestep_cutoff", MOSS_CUTOFF))
            if cutoff < 1.0:
                top = torch.special.ndtr(torch.tensor((torch.logit(torch.tensor(cutoff)).item() - mean) / std))
                p = torch.rand(1, device=self.device) * top.to(self.device)
                base = torch.sigmoid(mean + std * torch.special.ndtri(p)) / cutoff
            else:
                base = torch.sigmoid(torch.randn(1, device=self.device) * std + mean)
        elif distribution == "uniform":
            base = torch.rand(1, device=self.device)
        else:
            raise ValueError(f"train.timestep_distribution must be logit_normal or uniform, got {distribution!r}")
        mix = float(train_config.get("timestep_uniform_mix", MOSS_UNIFORM_MIX) or 0.0)
        if mix > 0 and torch.rand(()).item() < mix:
            base = torch.rand(1, device=self.device)
        shift = float(train_config.get("timestep_shift", 1.0))
        sigma = shift * base / (1.0 + (shift - 1.0) * base)
        grid = int(train_config.get("timestep_grid", MOSS_GRID) or 0)
        if grid > 0:
            sigma = torch.ceil(sigma * grid).clamp(1, grid) / grid
        lo = float(train_config.get("min_sigma", 0.001))
        hi = float(train_config.get("max_sigma", 1.0))
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
            # The empty prompt, exactly the negative prompt the reference sampler's CFG uses.
            caption = ""
        x0 = self.latent_cache[key].to(self.device, torch.float32).unsqueeze(0)
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)

        sigma = self._sample_sigma()
        self.last_sigma = sigma  # for the monitor's noise bands
        noise = torch.randn_like(x0)
        s = sigma.view(1, 1, 1, 1)
        noisy = (1.0 - s) * x0 + s * noise
        pred = self.lora_network(noisy.to(self.dtype), sigma, context)
        loss = torch.nn.functional.mse_loss(pred.float(), (noise - x0).float())
        strength = float(train_config.get("loss_weight_strength", 1.0))
        return loss * strength if strength != 1.0 else loss

    def train(self):
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
        self.run_metadata["resolved_sigma_shift"] = float(train_config.get("timestep_shift", 1.0))
        logger.info(
            "Starting MOSS-Image LoRA training: %d epochs, %d total steps; timesteps %s, shift %.2f",
            num_epochs,
            total_steps,
            train_config.get("timestep_distribution", "logit_normal"),
            self.run_metadata["resolved_sigma_shift"],
        )

        progress_bar = tqdm(total=total_steps, desc="Training Steps", disable=not self.is_main_process)
        keys = KeyCommands(menu=self._preview_menu) if self.is_main_process else None
        monitor = TrainMonitor(every=int(train_config.get("monitor_every", 100) or 100))
        stopped = False
        global_step = 0
        loss = None
        monitor.exclude(self._maybe_preview(0, total_steps))
        for epoch in range(num_epochs):
            self.lora_network.train()
            for index in self._epoch_order():
                with self.accelerator.accumulate(self.lora_network):
                    loss = self._training_step(index)
                    self.accelerator.backward(loss)
                    monitor.record(loss.item(), float(self.last_sigma))
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
                    monitor.step_done(global_step, total_steps, self.optimizer)
                    monitor.exclude(self._maybe_preview(global_step, total_steps))
                    if save_every > 0 and global_step % save_every == 0 and global_step < total_steps:
                        self.save_checkpoint(global_step)
                    stopped = keys is not None and keys.handle(global_step, self.save_checkpoint)
                    if keys is not None:
                        monitor.exclude(keys.take_paused_seconds())
                    self._watch_config()
                progress_bar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "step": global_step, "epoch": f"{epoch + 1}/{num_epochs}"}
                )
                if global_step >= total_steps or stopped:
                    break
            if global_step >= total_steps or stopped:
                break

        progress_bar.close()
        if keys is not None:
            keys.close()
        if stopped:
            logger.info(
                "MOSS-Image LoRA training stopped by key at step %d (saved as step %d).", global_step, global_step
            )
            return
        self.save_checkpoint("final")
        logger.info("MOSS-Image LoRA training completed!")

    def _log_step(self, step, total, loss, epoch, num_epochs):
        group = self.optimizer.param_groups[0]
        inner = getattr(self.optimizer, "optimizer", self.optimizer)
        prefix = (
            f"Step: {step}/{total} [{int(step / total * 100)}%], "
            f"Loss: {loss.item():.6f}, Epoch: {epoch + 1}/{num_epochs}, "
        )
        prefix += getattr(self, "step_note", "")
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

    def save_checkpoint(self, step):
        if not self.is_main_process:
            return
        output_config = self.config.get("output", {})
        output_dir = Path(output_config["path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        name = output_config.get("name") or output_config.get("lora_name") or "moss_image_lora"
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

    def _dry_run_samples(self) -> int:
        """One epoch's samples, counted from the dataset folder without decoding anything."""
        dataset_config = self.config["dataset"]
        repeats = max(int(dataset_config.get("repeats", 1) or 1), 1)
        return max(count_media_files(dataset_config.get("path", ""), ["image"]), 1) * repeats

    @classmethod
    def step_groups(cls, config: dict) -> list[Group]:
        """The image folders an epoch draws from, for --suggest-steps."""
        dataset_config = config["dataset"]
        root = dataset_config.get("path", "")
        return [Group("Subjects", root, float(dataset_config.get("repeats", 1) or 1), folder_counts(root))]

    @classmethod
    def step_notes(cls, config: dict) -> list[str]:  # noqa: ARG003 - overridden with the config in use
        return []

    @classmethod
    def suggest_steps(cls, config: dict, epochs: float) -> str:
        return step_report(cls.step_groups(config), config.get("train", {}), epochs, cls.step_notes(config))

    def simulate(self):
        """Plan from the real dataset folder and tick through the steps without loading models."""
        samples = self._dry_run_samples()
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
    parser = argparse.ArgumentParser(description="Train a MOSS-Image LoRA in maltrainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without loading models")
    parser.add_argument("--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step")
    parser.add_argument(
        "--suggest-steps",
        type=float,
        nargs="?",
        const=DEFAULT_EPOCHS,
        metavar="EPOCHS",
        help=f"Count the dataset's images and suggest train.max_steps for EPOCHS epochs "
        f"(default {DEFAULT_EPOCHS}), then exit without training",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.suggest_steps is not None:
        print(MOSSImageLoRATrainer.suggest_steps(config, args.suggest_steps))
        return
    setup_logging(config.get("logging", {}))
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])
    trainer = MOSSImageLoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time)
    trainer.config_path = Path(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
