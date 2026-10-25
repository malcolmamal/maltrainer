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
- as the MOSS-Image finetune: regularisation images, per-folder repeats, prior preservation and --suggest-steps
  (maltrainer/trainers/finetune/: dataset.py, prior_setup.py), previews and the gallery (trainers/krea2/previews.py)

Caching, timesteps and loss are the Krea 2 LoRA trainer's (train_krea2_lora.py).

    python train_krea2_finetune.py --config templates/krea2_finetune_template.yaml
    python train_krea2_finetune.py --config my_run.yaml --resume-from output/.../krea2_ft_x_state
    python train_krea2_finetune.py --config my_run.yaml --suggest-steps
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import torch

from maltrainer.finetune.rotation_base import original_weights
from maltrainer.optimizers import Automagic3
from maltrainer.trainers.finetune.prior_setup import FinetunePriorSetup
from maltrainer.trainers.krea2 import finetune_loop, finetune_optimizer, finetune_save
from maltrainer.trainers.krea2.finetune_host import STATE_PRIOR
from maltrainer.trainers.krea2.finetune_model import load_for_finetune
from maltrainer.trainers.krea2.finetune_optimizer import automagic_kwargs
from maltrainer.trainers.krea2.previews import Krea2Previews
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.utils.config import load_config
from maltrainer.utils.step_planner import DEFAULT_EPOCHS
from train_krea2_lora import Krea2LoRATrainer

logger = logging.getLogger(__name__)


class Krea2FinetuneTrainer(FinetunePriorSetup, Krea2Previews, Krea2LoRATrainer):
    prior_state_name = STATE_PRIOR
    rotator: Any  # the window rotator and the always-trained linears, set by load_transformer_and_lora
    always_on: list[tuple[str, torch.nn.Parameter]]

    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        if config.get("train", {}).get("dop"):
            raise ValueError(
                "train.dop is the LoRA trainer's; a finetune holds its subjects to the original model with "
                "prior.weight (docs/krea2-finetuning.md)"
            )
        self._init_previews(config)
        self._check_preview_size()
        super().__init__(config, dry_run=dry_run, dry_run_step_time=dry_run_step_time)
        self.run_metadata["model_type"] = "krea2_finetune"
        if self.accelerator.num_processes > 1:
            raise ValueError("Krea 2 finetuning runs on one GPU")
        if int(config.get("train", {}).get("gradient_accumulation_steps", 1) or 1) != 1:
            raise ValueError("fused Automagic3 steps inside backward, so gradient_accumulation_steps must be 1")
        self._rotate_every()  # a bad window length stops the run here, before the model is loaded
        resume = config.get("train", {}).get("resume_from")
        self.resume_dir = Path(resume) if resume else None
        self.resume_state: dict[str, Any] | None = None
        self.global_step = 0
        self.epoch = 0
        self.position = 0
        self.epoch_rng: tuple[object, ...] | None = None
        self.always_optimizer: Automagic3 | None = None
        self.key_prefix = ""
        self.step_note = ""  # the step's prior term, on the log line
        # dataset.reg_path, prior.weight > 0 and the gallery (maltrainer/trainers/finetune/dataset.py)
        self._init_finetune_data(config)

    def load_transformer_and_lora(self) -> None:
        """The DiT with its window rotator and always-trained linears (maltrainer.trainers.krea2.finetune_model)."""
        load_for_finetune(self, lambda text: print(text, flush=True))

    def _rotate_every(self) -> int:
        """``finetune.rotate_every`` steps per window; left out or null it is 100, below 1 it is an error."""
        value = (self.config.get("finetune", {}) or {}).get("rotate_every")
        every = 100 if value is None else int(value)
        if every < 1:
            raise ValueError(f"finetune.rotate_every must be at least 1 step (leave it out for 100), got {value}")
        return every

    def _make_optimizer(self, params: list[torch.nn.Parameter], name: str) -> Automagic3:
        return Automagic3([{"params": params, "name": name}], **automagic_kwargs(self.config["train"]))

    def setup_optimizer(self) -> None:
        """Fused Automagic3 only, and the saved states on resume (maltrainer.trainers.krea2.finetune_optimizer)."""
        finetune_optimizer.setup_optimizer(self)

    def _window_at(self, step: int) -> int:
        return (step // self._rotate_every()) % len(self.rotator.windows)

    def _prior_latent(self, key: str) -> torch.Tensor:
        return self.latent_cache[key][0]  # cached as [1, C, H, W]

    def _prior_sigma(self, x0: torch.Tensor) -> torch.Tensor:
        """A training sigma for the image, shifted by its token count as in ``_training_step``."""
        return self._sample_sigma((x0.shape[-2] // 2) * (x0.shape[-1] // 2))[0]

    def _original_weights(self) -> contextlib.AbstractContextManager[Any]:
        """The base checkpoint's values in the rotating and always-on weights (maltrainer/finetune/rotation_base.py)."""
        return original_weights(self.rotator, self.always_on, self._original_file())

    def train(self) -> None:
        """Windows, fused steps, saves, keys and an exact resume (maltrainer.trainers.krea2.finetune_loop)."""
        finetune_loop.train(self)

    def _log_step(self, step: int, total: int, loss: torch.Tensor, epoch: int, num_epochs: int) -> None:
        window = self.rotator.names[self.rotator.current] if self.rotator.current is not None else "-"
        self.progress_logger.info(
            "Step: %d/%d [%d%%], Loss: %.6f, Epoch: %d/%d, %sWindow: %s, LR: %.2e",
            step,
            total,
            int(step / total * 100),
            loss.item(),
            epoch + 1,
            num_epochs,
            self.step_note,
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
    parser.add_argument(
        "--suggest-steps",
        type=float,
        nargs="?",
        const=DEFAULT_EPOCHS,
        metavar="EPOCHS",
        help=f"Count the subject and regularisation images and suggest train.max_steps for "
        f"EPOCHS epochs of each (default {DEFAULT_EPOCHS}), then exit without training",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.suggest_steps is not None:
        print(Krea2FinetuneTrainer.suggest_steps(config, args.suggest_steps))
        return
    if args.resume_from:
        config.setdefault("train", {})["resume_from"] = args.resume_from
    start_run(config)
    trainer = Krea2FinetuneTrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time)
    trainer.config_path = Path(args.config)  # its sample: section is reloaded when the file is saved
    trainer.run()


if __name__ == "__main__":
    main()
