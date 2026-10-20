#!/usr/bin/env python3
"""
Krea 2 full finetuning for maltrainer: the DiT's own weights, on one 32 GB card.

Krea 2's 12.9B bf16 weights (25.8 GB) do not fit beside their training state, so the blocks
train a window at a time (maltrainer/finetune/rotation.py, after Fizgig):

- the frozen model stays on the GPU in fp8 (block Linears per output row), as for LoRAs
- one window (every block's attention, then every block's MLP gate, up, down) is trainable in
  bf16 from a CPU master copy; ``finetune.rotate_every`` steps later it is written back and the
  next window comes up. The text-fusion stack trains all run
- fused Automagic3 steps each weight inside backward (no full gradient copy) and rounds the
  update stochastically into bf16. Each window keeps its optimizer state in CPU RAM while frozen
- saves write a full bf16 ComfyUI file (~26 GB) plus a resume state

Data, caching, timesteps and loss are the Krea 2 LoRA trainer's (train_krea2_lora.py).

    python train_krea2_finetune.py --config templates/krea2_finetune_template.yaml
    python train_krea2_finetune.py --config my_run.yaml --resume-from output/.../krea2_ft_x_state
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import torch

from maltrainer.optimizers import Automagic3
from maltrainer.trainers.krea2 import finetune_loop, finetune_save
from maltrainer.trainers.krea2.finetune_host import STATE_OPTIMIZER, STATE_TRAINER
from maltrainer.trainers.krea2.finetune_model import load_for_finetune
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.utils.config import load_config
from train_krea2_lora import Krea2LoRATrainer

logger = logging.getLogger(__name__)


class Krea2FinetuneTrainer(Krea2LoRATrainer):
    rotator: Any  # the window rotator and the always-trained linears, set by load_transformer_and_lora
    always_on: list[tuple[str, torch.nn.Parameter]]

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        if config.get("train", {}).get("dop"):
            raise ValueError("train.dop is the LoRA trainer's; the Krea 2 finetune has no prior preservation yet")
        super().__init__(config, dry_run=dry_run, dry_run_step_time=dry_run_step_time)
        self.run_metadata["model_type"] = "krea2_finetune"
        if self.accelerator.num_processes > 1:
            raise ValueError("Krea 2 finetuning runs on one GPU")
        if int(config.get("train", {}).get("gradient_accumulation_steps", 1) or 1) != 1:
            raise ValueError("fused Automagic3 steps inside backward, so gradient_accumulation_steps must be 1")
        resume = config.get("train", {}).get("resume_from")
        self.resume_dir = Path(resume) if resume else None
        self.resume_state: dict[str, Any] | None = None
        self.global_step = 0
        self.epoch = 0
        self.position = 0
        self.epoch_rng: tuple[object, ...] | None = None
        self.always_optimizer: Automagic3 | None = None
        self.key_prefix = ""

    def load_transformer_and_lora(self) -> None:
        """The DiT with its window rotator and always-trained linears (maltrainer.trainers.krea2.finetune_model)."""
        load_for_finetune(self, lambda text: print(text, flush=True))

    def _rotate_every(self) -> int:
        return max(int((self.config.get("finetune", {}) or {}).get("rotate_every", 100) or 100), 1)

    def _automagic_kwargs(self) -> dict[str, Any]:
        train_config = self.config["train"]
        return {
            "lr": float(train_config.get("learning_rate", 1e-5)),
            "min_lr": float(train_config.get("automagic_min_lr", 1e-5)),
            "max_lr": float(train_config.get("automagic_max_lr", 1e-5)),
            "beta2": float(train_config.get("automagic_beta2", 0.999)),
            "eps": float(train_config.get("automagic_eps", 1e-30)),
            "clip_threshold": float(train_config.get("automagic_clip_threshold", 1.0)),
            "weight_decay": float(train_config.get("weight_decay", 0.0)),
            "polarity_history": int(train_config.get("automagic_polarity_history", 8)),
            "fused": True,
        }

    def _make_optimizer(self, params: list[torch.nn.Parameter], name: str) -> Automagic3:
        return Automagic3([{"params": params, "name": name}], **self._automagic_kwargs())

    def setup_optimizer(self) -> None:
        name = str(self.config["train"].get("optimizer", "automagic3")).lower()
        if name not in {"automagic3", "automagic", "automagic_v3"}:
            raise ValueError(
                f"Krea 2 finetuning supports automagic3 only (got {name!r}): it rounds updates stochastically "
                "into the bf16 weights and steps inside backward"
            )
        self.run_metadata["optimizer_resolved"] = {"name": "automagic3", **self._automagic_kwargs()}
        logger.info("Optimizer: %s", self.run_metadata["optimizer_resolved"])
        if self.always_on:
            self.always_optimizer = self._make_optimizer([p for _, p in self.always_on], "txtfusion")
        if self.resume_dir is not None:
            saved = torch.load(self.resume_dir / STATE_OPTIMIZER, map_location="cpu", weights_only=False)
            self.rotator.load_state_dict(saved["rotation"])
            if self.always_optimizer is not None and saved.get("always") is not None:
                self.always_optimizer.load_state_dict(saved["always"])
            self.resume_state = torch.load(self.resume_dir / STATE_TRAINER, map_location="cpu", weights_only=False)

    def _window_at(self, step: int) -> int:
        return (step // self._rotate_every()) % len(self.rotator.windows)

    def _dataset_fingerprint(self) -> str:
        root = self.config["dataset"]["path"]
        digest = hashlib.sha1(str(self.config["dataset"].get("repeats", 1)).encode("utf-8"), usedforsecurity=False)
        for key, caption in self.samples:
            digest.update(f"\x1e{Path(os.path.relpath(key, root)).as_posix()}\x1f{caption}".encode())
        return digest.hexdigest()

    def train(self) -> None:
        """Windows, fused steps, saves, keys and an exact resume (maltrainer.trainers.krea2.finetune_loop)."""
        finetune_loop.train(self)

    def _log_step(self, step: int, total: int, loss: torch.Tensor, epoch: int, num_epochs: int) -> None:
        window = self.rotator.names[self.rotator.current] if self.rotator.current is not None else "-"
        self.progress_logger.info(
            "Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, Window: %s, LR: %.2e",
            step,
            total,
            int(step / total * 100),
            loss.item(),
            epoch + 1,
            num_epochs,
            window,
            self.rotator.optimizer.get_avg_learning_rate(),
        )

    def _name(self) -> str:
        name = self.config.get("output", {}).get("name") or "krea2_finetune"
        return name[: -len(".safetensors")] if name.endswith(".safetensors") else name

    def _model_bytes(self) -> int:
        return sum(t.numel() * 2 for t in self.transformer.state_dict().values())  # bf16, an upper bound

    def save_checkpoint(self, step: int | str) -> bool:
        """The full model and the resume state (maltrainer.trainers.krea2.finetune_save)."""
        return finetune_save.save_checkpoint(self, step)

    def _save_state(self, state_dir: Path, weights: Path) -> None:
        finetune_save.save_state(self, state_dir, weights)


def main() -> None:
    parser = training_arguments("Finetune the Krea 2 DiT in maltrainer")
    parser.add_argument("--resume-from", type=str, help="A <name>_state folder; overrides train.resume_from")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.resume_from:
        config.setdefault("train", {})["resume_from"] = args.resume_from
    start_run(config)
    Krea2FinetuneTrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time).run()


if __name__ == "__main__":
    main()
