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

import logging
from pathlib import Path
from typing import Any

import torch

from maltrainer.control.launch import make_preview_host_keys
from maltrainer.models.moss_image.loading import load_moss_image_transformer
from maltrainer.models.moss_image.lora import DEFAULT_SKIP_BLOCKS, MOSSImageLoRANetwork
from maltrainer.trainers.moss_image.advice import StepAdvice
from maltrainer.trainers.moss_image.cache import cache_latents, cache_prompts, collect_samples, prompt_folders
from maltrainer.trainers.moss_image.previews import PreviewHooks
from maltrainer.trainers.moss_image.timesteps import sample_sigma
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.training.image_lora import ImageLoRATrainer
from maltrainer.utils.compile import compile_blocks
from maltrainer.utils.config import load_config
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.step_planner import DEFAULT_EPOCHS
from maltrainer.utils.train_monitor import TrainMonitor

logger = logging.getLogger(__name__)


class MOSSImageLoRATrainer(PreviewHooks, StepAdvice, ImageLoRATrainer):
    model_type = "moss_image"
    label = "MOSS-Image"
    default_name = "moss_image_lora"
    batch_one_reason = (
        "MOSS-Image trains at batch_size 1 (every caption is encoded at its own length, and "
        "the RoPE follows it, as at inference); use train.gradient_accumulation_steps for a "
        "larger effective batch."
    )
    # Attention, MLP and I/O each adapt their own rate. The sign window is 8 steps: on FLUX.3 a
    # 16-step window left the rate stuck near 3e-5 with batch-1 gradients.
    polarity_history = 8
    strict_optimizer = True
    schedule = ("logit_normal", 1.0)

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025) -> None:
        self._init_previews(config)  # first: a sample: warning is logged before the device lines, as before
        super().__init__(config, dry_run=dry_run, dry_run_step_time=dry_run_step_time)

    # ------------------------------------------------------------------ caching

    def precache_dataset(self) -> None:
        dataset_config = self.config["dataset"]
        sources = self._sources()
        if not sources:
            raise ValueError(f"No images found under {dataset_config['path']}")
        print("=== Step 1/2: Pre-caching Dataset (VAE Latents & Text Embeddings) ===", flush=True)
        latents, encoded = cache_latents(sources, self.config, self.device, self.is_main_process)
        sizes = collect_samples(sources, latents, self.samples, self.latent_cache)
        logger.info(
            "Training sizes (w x h: images): %s", ", ".join(f"{w}x{h}: {n}" for (w, h), n in sorted(sizes.items()))
        )
        logger.info("Latents: %d from the disk cache, %d encoded", len(sources) - encoded, encoded)
        print("VAE latents ready.", flush=True)

        dropout = float(self.config.get("train", {}).get("caption_dropout", 0.0) or 0.0) > 0
        folders = prompt_folders(self.samples, dataset_config["path"], self._extra_prompts(), dropout)
        self.text_cache = cache_prompts(folders, self.config, (self.device, self.dtype), self.is_main_process)
        print("Text embeddings ready.", flush=True)
        logger.info(
            "Pre-cached %d images and %d prompt embeddings (text lengths: %s tokens).",
            len(self.samples),
            len(self.text_cache),
            sorted({embeds.shape[0] for embeds in self.text_cache.values()}),
        )

    # ------------------------------------------------------------------ model

    def load_transformer_and_lora(self) -> None:
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
        self._lora_ready()
        self.run_metadata["lora_skip_blocks"] = list(self.lora_network.skip_blocks)
        self._maybe_compile()

    def _maybe_compile(self) -> None:
        """train.compile: torch.compile each of the 42 blocks (after the LoRA wraps their Linears)."""
        train_config = self.config.get("train", {})
        if not bool(train_config.get("compile", False)):
            self.run_metadata["compiled"] = False
            return
        self.run_metadata["compiled"] = compile_blocks(
            self.transformer.layers, self.device, mode=train_config.get("compile_mode") or None
        )

    # ------------------------------------------------------------------ training

    def _sample_sigma(self) -> torch.Tensor:
        """Flow-matching sigma for one sample (maltrainer.trainers.moss_image.timesteps)."""
        return sample_sigma(self.config["train"], self.device)

    def _training_step(self, index: int) -> torch.Tensor:
        # caption dropout trains on the empty prompt, exactly the negative prompt the reference sampler's CFG uses
        key, caption = self._sample(index)
        x0 = self.latent_cache[key].to(self.device, torch.float32).unsqueeze(0)
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)

        sigma = self._sample_sigma()
        self.last_sigma = sigma  # for the monitor's noise bands
        noise = torch.randn_like(x0)
        s = sigma.view(1, 1, 1, 1)
        noisy = (1.0 - s) * x0 + s * noise
        pred = self.lora_network(noisy.to(self.dtype), sigma, context)
        loss = torch.nn.functional.mse_loss(pred.float(), (noise - x0).float())
        strength = float(self.config["train"].get("loss_weight_strength", 1.0))
        return loss * strength if strength != 1.0 else loss

    # The loop's hooks: key e opens the preview menu; the monitor (train.monitor_every) records every sample, its
    # speed leaves out previews and pauses; previews and the sample: reload run at the step boundaries.

    def _make_keys(self) -> KeyCommands | None:
        return make_preview_host_keys(KeyCommands, self)

    def _before_loop(self, total_steps: int) -> None:
        self.monitor = TrainMonitor(every=int(self.config["train"].get("monitor_every", 100) or 100))
        self.monitor.exclude(self._maybe_preview(0, total_steps))

    def _micro_step(self, index: int) -> torch.Tensor:
        loss = super()._micro_step(index)
        self.monitor.record(loss.item(), float(self.last_sigma))
        return loss

    def _after_log(self, step: int, total: int) -> None:
        self.monitor.step_done(step, total, self.optimizer)
        self.monitor.exclude(self._maybe_preview(step, total))

    def _after_keys(self, keys: KeyCommands | None) -> None:
        if keys is not None:
            self.monitor.exclude(keys.take_paused_seconds())
        self._watch_config()


def main() -> None:
    parser = training_arguments("Train a MOSS-Image LoRA in maltrainer")
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
    start_run(config)
    trainer = MOSSImageLoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time)
    trainer.config_path = Path(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
