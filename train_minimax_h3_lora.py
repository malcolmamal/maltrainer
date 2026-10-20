#!/usr/bin/env python3
"""
MiniMax H3 LoRA Training Script for Maltrainer.
Trains LoRA adapters for MiniMax H3 using INT8 ConvRot transformer, FP16 Video VAE, FP32 audio
VAE and Qwen3-VL text encoder, on stills, video clips with sound, and audio-only recordings.
Uses automated dataset pre-caching for 40x step acceleration (~0.8s/step).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import set_seed

from maltrainer.models.minimax_h3.packing import packed_layout
from maltrainer.trainers.minimax_h3 import dry_run, loop
from maltrainer.trainers.minimax_h3.data import count_dataset_samples, precache
from maltrainer.trainers.minimax_h3.model import add_lora, load_transformer
from maltrainer.trainers.minimax_h3.optimizer import build_optimizer, warmup_scheduler
from maltrainer.trainers.minimax_h3.sample import H3Sample
from maltrainer.trainers.minimax_h3.step import guided_target, training_step
from maltrainer.trainers.minimax_h3.timesteps import resolve_flow_shift, sample_timesteps
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.training.saving import write_lora
from maltrainer.training.session import log_device, make_accelerator, open_training_log, start_metadata
from maltrainer.training.steps import epoch_order
from maltrainer.utils.config import load_config
from maltrainer.utils.ema import ParameterEMA

logger = logging.getLogger(__name__)


def _say(text: str) -> None:
    print(text, flush=True)


class MiniMaxH3LoRATrainer:
    """Trainer for MiniMax H3 LoRA models."""

    _guided_target = staticmethod(guided_target)

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        self.config = config
        self.dry_run, self.dry_run_step_time = dry_run, dry_run_step_time
        self.last_saved_path: str | None = None
        self.lr_scheduler: Any = None
        # Resolved facts about this run, embedded into every checkpoint's metadata. Populated
        # as they become known, since several are only decided at load/first-step time.
        self.run_metadata = start_metadata(None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.samples: list[H3Sample] = []
        self._layout_cache: dict[tuple[Any, ...], dict[str, torch.Tensor]] = {}
        self.text_cache: dict[str, torch.Tensor] = {}
        self.ema: ParameterEMA | None = None
        if int(config.get("train", {}).get("batch_size", 1) or 1) != 1:
            # Every item packs its own sequence (text length, frame count, aspect and audio
            # length all vary) and H3's per-timestep modulation is shared across the batch.
            raise ValueError(
                "MiniMax H3 trains at batch_size 1; use train.gradient_accumulation_steps for a larger effective batch."
            )
        self.accelerator = make_accelerator(config)
        self.is_main_process = self.accelerator.is_main_process
        if self.is_main_process:
            log_device(self.device)
        self.progress_logger = open_training_log(config)

    def _train_audio(self) -> bool:
        return bool(self.config.get("train", {}).get("train_audio", True))

    def _do_guidance_loss(self) -> bool:
        train_config = self.config.get("train", {})
        return bool(train_config.get("do_guidance_loss", train_config.get("do_differential_guidance", False)))

    def precache_dataset(self) -> None:
        """Encode every dataset item once (maltrainer.trainers.minimax_h3.data)."""
        flags = (self._train_audio(), self._do_guidance_loss())
        self.text_cache, counts = precache(
            self.config, self.samples, flags, self.device, self.dtype, _say, self.is_main_process
        )
        self.run_metadata["dataset_composition"] = counts

    def load_transformer_and_lora(self) -> None:
        """Load INT8 ConvRot transformer base and inject LoRA network."""
        print("=== Step 2/2: Loading MiniMax H3 INT8 Transformer Base ===", flush=True)
        logger.info("Loading Transformer base and LoRA network...")
        self.transformer = load_transformer(self.config, self.device, self.run_metadata, _say)
        on = (self.device, self.dtype)
        self.lora_network, self.lora_dtype = add_lora(self.transformer, self.config, on, self.run_metadata)
        print("Transformer and LoRA network ready on GPU!", flush=True)

    def _epoch_order(self) -> list[int]:
        """Shuffled sample indices for one epoch (each sample ``repeats`` times), this rank's share."""
        repeats = self.config["dataset"].get("repeats", 1)
        return epoch_order(len(self.samples), repeats, self.accelerator.num_processes, self.accelerator.process_index)

    def _count_dataset_samples(self) -> int:
        return count_dataset_samples(self.config.get("dataset", {}))

    def dry_run_training(self) -> None:
        """Simulate training without downloading or loading models (maltrainer.trainers.minimax_h3.dry_run)."""
        samples = self._count_dataset_samples()
        self.last_saved_path = dry_run.simulate(self.config, samples, self.dry_run_step_time, self.is_main_process)

    def _build_packed_layout(
        self, num_text_tokens: int, latent_t: int, latent_h: int, latent_w: int, audio_t: int = 0
    ) -> dict[str, torch.Tensor]:
        """maltrainer.models.minimax_h3.packing.packed_layout, cached per shape signature."""
        key = (num_text_tokens, latent_t, latent_h, latent_w, audio_t)
        if key not in self._layout_cache:
            self._layout_cache[key] = packed_layout(num_text_tokens, latent_t, latent_h, latent_w, audio_t, self.device)
        return self._layout_cache[key]

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return sample_timesteps(self.config.get("train", {}), batch_size, self.device)

    def _resolve_flow_shift(self, latent_h: int, latent_w: int) -> float:
        return resolve_flow_shift(self.config.get("train", {}), latent_h, latent_w)

    def setup_optimizer(self) -> None:
        """Setup optimizer over the LoRA adapters only (maltrainer.trainers.minimax_h3.optimizer)."""
        train_config = self.config["train"]
        params = list(self.lora_network.trainable_parameters())
        self.optimizer, resolved = build_optimizer(params, train_config, self.lora_dtype)
        if resolved is not None:
            self.run_metadata["optimizer_resolved"] = resolved
        self.lr_scheduler = warmup_scheduler(self.optimizer, train_config)

    def _forward(
        self, video_patches: Any, audio_rows: Any, text_embeds: Any, timestep: Any, layout: dict[str, Any]
    ) -> Any:
        return self.lora_network(
            video_hidden_states=video_patches,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=text_embeds,
            timestep=timestep,
            timestep_indices=layout["timestep_indices"],
            token_tags=layout["token_tags"],
            position_ids=layout["position_ids"],
            video_indices=layout["video_indices"],
            audio_indices=layout["audio_indices"],
            text_indices=layout["text_indices"],
        )

    def _training_step(self, sample: H3Sample, flow_shift: float) -> tuple[torch.Tensor, dict[str, float]]:
        """Loss for one cached sample (maltrainer.trainers.minimax_h3.step). Returns (loss, parts for logging)."""
        return training_step(self, sample, flow_shift)

    def train(self) -> None:
        """Main MiniMax H3 LoRA training loop (maltrainer.trainers.minimax_h3.loop)."""
        loop.train(self)

    def save_checkpoint(self, step: int | str) -> None:
        """Save LoRA checkpoint safetensors file (checkpoints carry the EMA average; training goes on live)."""
        if not self.is_main_process:
            return
        save_path = Path(self.config["output"]["path"]) / f"minimax_h3_lora_step_{step}.safetensors"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dtype = getattr(torch, str(self.config.get("output", {}).get("dtype", "bfloat16")))
        network = self.accelerator.unwrap_model(self.lora_network)
        if self.ema is not None:
            with self.ema.applied():
                state_dict = network.get_state_dict(save_dtype=save_dtype)
        else:
            state_dict = network.get_state_dict(save_dtype=save_dtype)
        write_lora(self.config, state_dict, save_path, step, self.run_metadata, save_dtype)
        self.last_saved_path = str(save_path)
        logger.info("Saved LoRA checkpoint to %s", save_path)

    def run(self) -> None:
        """Run training pipeline."""
        try:
            set_seed(self.config.get("seed", 42))
            if self.dry_run:
                self.dry_run_training()
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
            raise e


def main() -> None:
    args = training_arguments("Train MiniMax H3 LoRA in Maltrainer").parse_args()
    config = load_config(args.config)
    start_run(config)
    MiniMaxH3LoRATrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
