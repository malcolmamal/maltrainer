#!/usr/bin/env python3
"""
MiniMax H3 Image full finetuning for maltrainer: the DiT's own weights, on one 32 GB card, from ComfyUI's INT8 file.

The 19.3B INT8 weights (21 GB) do not fit beside their bf16 training state, so the blocks train a window at a time
(maltrainer/finetune/rotation_int8.py, after Fizgig's H3 finetune):

- the frozen model stays on the GPU as INT8 ConvRot codes, as for LoRAs
- one window (``finetune.blocks_per_window`` blocks) is trainable in bf16; ``finetune.rotate_every`` steps later it is
  quantized back to INT8 and the next window comes up. Each window's bf16 value is kept in CPU RAM
- fused Automagic3 steps each weight inside backward and rounds the update stochastically into bf16
- Fizgig's training adapter (``model.training_adapter_path``) rides frozen over the Linears while training unless
  ``finetune.training_adapter`` is false; it is not in the saved model
- saves write an INT8 ConvRot file ComfyUI loads like the original, plus a resume state

Caching, noise levels and loss are the H3 Image LoRA trainer's (train_minimax_h3_image_lora.py).

    python train_minimax_h3_image_finetune.py --config templates/minimax_h3_image_finetune_template.yaml
    python train_minimax_h3_image_finetune.py --config my_run.yaml --resume-from output/.../<name>_state
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import torch

from maltrainer.finetune.resume import load_resume_file
from maltrainer.optimizers import Automagic3
from maltrainer.trainers.krea2.finetune_optimizer import automagic_kwargs
from maltrainer.trainers.minimax_h3_image import finetune_loop, finetune_model, finetune_save
from maltrainer.trainers.minimax_h3_image.finetune_host import STATE_OPTIMIZER, STATE_TRAINER
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.utils.config import load_config
from train_minimax_h3_image_lora import MiniMaxH3ImageTrainer
from train_minimax_h3_lora import _say

logger = logging.getLogger(__name__)


class MiniMaxH3ImageFinetuneTrainer(MiniMaxH3ImageTrainer):
    checkpoint_prefix, label = "minimax_h3_image_finetune", "MiniMax H3 Image finetune"
    rotator: Any  # the window rotator and the always-trained weights, set by load_transformer_and_lora
    always_on: list[tuple[str, torch.nn.Parameter]]
    adapter_layers: Any  # the frozen training adapter's layers when it is on

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        if int(config.get("train", {}).get("gradient_accumulation_steps", 1) or 1) != 1:
            raise ValueError("fused Automagic3 steps inside backward, so gradient_accumulation_steps must be 1")
        super().__init__(config, dry_run=dry_run, dry_run_step_time=dry_run_step_time)
        self.run_metadata["model_type"] = "minimax_h3_image_finetune"
        if self.accelerator.num_processes > 1:
            raise ValueError("MiniMax H3 Image finetuning runs on one GPU")
        self._rotate_every()  # a bad window length stops the run here, before the model is loaded
        resume = config.get("train", {}).get("resume_from")
        self.resume_dir = Path(resume) if resume else None
        self.resume_state: dict[str, Any] | None = None
        self.global_step = self.epoch = self.position = 0
        self.epoch_rng: tuple[object, ...] | None = None
        self.always_optimizer: Automagic3 | None = None

    def load_transformer_and_lora(self) -> None:
        """The INT8 DiT with its window rotator and always-trained weights (finetune_model.py)."""
        finetune_model.load_for_finetune(self, _say)

    def _rotate_every(self) -> int:
        """``finetune.rotate_every`` steps per window; left out or null it is 100, below 1 it is an error."""
        value = (self.config.get("finetune", {}) or {}).get("rotate_every")
        every = 100 if value is None else int(value)
        if every < 1:
            raise ValueError(f"finetune.rotate_every must be at least 1 step (leave it out for 100), got {value}")
        return every

    def _window_at(self, step: int) -> int:
        return (step // self._rotate_every()) % len(self.rotator.windows)

    def _make_optimizer(self, params: list[torch.nn.Parameter], name: str) -> Automagic3:
        return Automagic3([{"params": params, "name": name}], **automagic_kwargs(self.config["train"]))

    def setup_optimizer(self) -> None:
        """Fused Automagic3 only; on resume, the saved states and counters."""
        name = str(self.config["train"].get("optimizer", "automagic3")).lower()
        if name not in ("automagic3", "automagic", "automagic_v3"):
            raise ValueError(
                f"MiniMax H3 Image finetuning supports automagic3 only (got {name!r}): it rounds updates "
                "stochastically into the bf16 weights and steps inside backward"
            )
        self.run_metadata["optimizer_resolved"] = {"name": "automagic3", **automagic_kwargs(self.config["train"])}
        if self.always_on:
            self.always_optimizer = self._make_optimizer([p for _, p in self.always_on], "always_on")
        if self.resume_dir is not None:
            saved = load_resume_file(self.resume_dir / STATE_OPTIMIZER, "cpu")
            self.rotator.load_state_dict(saved["rotation"])
            if self.always_optimizer is not None and saved.get("always") is not None:
                self.always_optimizer.load_state_dict(saved["always"])
            self.resume_state = load_resume_file(self.resume_dir / STATE_TRAINER, "cpu")

    def _epoch_size(self) -> int:
        """One epoch's samples, without drawing its order (so without touching Python's random)."""
        return len(self.samples) * max(int(self.config["dataset"].get("repeats", 1) or 1), 1)

    def _dataset_fingerprint(self) -> str:
        """The samples (paths relative to the dataset folder, captions) and repeats; a moved dataset still matches."""
        root = self.config["dataset"]["path"]
        digest = hashlib.sha1(str(self.config["dataset"].get("repeats", 1)).encode(), usedforsecurity=False)
        for sample in self.samples:
            digest.update(f"\x1e{Path(os.path.relpath(sample.key, root)).as_posix()}\x1f{sample.caption}".encode())
        return digest.hexdigest()

    def _name(self) -> str:
        name = self.config.get("output", {}).get("name") or "minimax_h3_image_finetune"
        return name[: -len(".safetensors")] if name.endswith(".safetensors") else name

    def train(self) -> None:
        finetune_loop.train(self)

    def save_checkpoint(self, step: int | str) -> None:
        finetune_save.save_checkpoint(self, step)

    def _save_state(self, state_dir: Path) -> None:
        finetune_save.save_state(self, state_dir)


def main() -> None:
    parser = training_arguments("Finetune the MiniMax H3 Image DiT in maltrainer")
    parser.add_argument("--resume-from", type=str, help="A <name>_state folder; overrides train.resume_from")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume_from:
        config.setdefault("train", {})["resume_from"] = args.resume_from
    start_run(config)
    trainer = MiniMaxH3ImageFinetuneTrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time)
    trainer.run()


if __name__ == "__main__":
    main()
