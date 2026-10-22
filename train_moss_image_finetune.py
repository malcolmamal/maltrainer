#!/usr/bin/env python3
"""
MOSS-Image full finetuning for maltrainer.

Trains the DiT's own weights (all 5.08B, or the subset finetune.trainable_prefixes names)
instead of a LoRA. Data, caching, timesteps and loss are the LoRA trainer's
(train_moss_image_lora.py); what differs is the model side:

- the bf16 DiT trains in place, its updates stochastically rounded into bf16 by Automagic3
- Automagic3 runs fused by default: each parameter steps inside backward the moment its gradient
  is ready, so no full set of gradients is ever held. Measured on an RTX 5090: 16.8 GB peak at
  1024 px, 19.0 GB at 1536 px. Fused means no gradient accumulation and no global gradient
  clipping; Automagic3's per-tensor RMS clip (automagic_clip_threshold) bounds each update.
- the optional EMA keeps its fp32 average (20 GB) in CPU RAM
- checkpoints are full ComfyUI single files in the source file's layout (~10 GB each)
- each save also writes <name>_state/ (raw weights, optimizer, EMA, step and RNG), which
  train.resume_from continues from exactly. Only the latest state is kept.
- by default only the Linear weights of the 42 joint blocks train (finetune.trainable_prefixes,
  finetune.linears_only); the text side stays frozen
- optional prior preservation (prior.weight, DOP) and regularisation images (dataset.reg_path),
  from maltrainer/finetune/

The parts are in maltrainer/trainers/moss_image/: finetune_data (regularisation, repeats), finetune_prior
(gallery, prior), finetune_state and finetune_checkpoint (saves), finetune_model, finetune_loop.

    python train_moss_image_finetune.py --config templates/moss_image_finetune_template.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from maltrainer.finetune.prior import PriorPreservation, resolve_classes
from maltrainer.trainers.moss_image import finetune_loop, finetune_model
from maltrainer.trainers.moss_image.finetune_checkpoint import FinetuneCheckpoint
from maltrainer.training.cli import start_run, training_arguments
from maltrainer.utils.config import load_config
from maltrainer.utils.ema import OffloadedParameterEMA
from maltrainer.utils.gallery import Gallery
from maltrainer.utils.step_planner import DEFAULT_EPOCHS
from train_moss_image_lora import MOSSImageLoRATrainer

logger = logging.getLogger(__name__)


class MOSSImageFinetuneTrainer(FinetuneCheckpoint, MOSSImageLoRATrainer):
    def __init__(self, config: dict[str, Any], dry_run: bool = False, dry_run_step_time: float = 0.025):
        super().__init__(config, dry_run=dry_run, dry_run_step_time=dry_run_step_time)
        self.run_metadata["model_type"] = "moss_image_finetune"
        if self.accelerator.num_processes > 1:
            raise ValueError("MOSS-Image finetuning runs on one GPU")
        self.ema: OffloadedParameterEMA | None = None
        resume = config.get("train", {}).get("resume_from")
        self.resume_dir = Path(resume) if resume else None
        self.resume_state: dict[str, Any] | None = None
        # Where the loop stands, for the resume state: the epoch, the next index into its
        # shuffled order, and the Python RNG state that shuffled it.
        self.global_step = 0
        self.epoch = 0
        self.position = 0
        self.epoch_rng: tuple[Any, ...] | None = None
        # dataset.reg_path: regularisation images, trained at train.reg_lr_multiplier x the step
        self.reg_keys: set[str] = set()
        self.reg_multiplier = float(config.get("train", {}).get("reg_lr_multiplier", 0.2))
        # prior.weight > 0: prior preservation against the original model (maltrainer/finetune/prior.py)
        prior_config = config.get("prior", {}) or {}
        self.prior: PriorPreservation | None = None
        self.prior_classes = {}
        self.prior_path: Path | None = None
        if float(prior_config.get("weight", 0.0) or 0.0) > 0:
            self.prior_classes = resolve_classes(prior_config, config.get("dataset", {}).get("trigger"))
            self.prior = PriorPreservation(
                weight=float(prior_config["weight"]),
                samples_per_image=int(prior_config.get("samples_per_image", 8)),
                every=int(prior_config.get("every", 1)),
                seed=int(config.get("seed", 42)),
            )
        self.gallery = Gallery.from_config(config, self.prior_classes)  # key g (maltrainer/utils/gallery.py)

    def load_transformer_and_lora(self) -> None:
        """The bf16 DiT and the trainable subset (maltrainer.trainers.moss_image.finetune_model)."""
        finetune_model.load_for_finetune(self, lambda text: print(text, flush=True))

    def setup_optimizer(self) -> None:
        """Fused Automagic3 over the trainable subset (maltrainer.trainers.moss_image.finetune_model)."""
        finetune_model.setup_optimizer(self)

    def train(self) -> None:
        """Steps, prior, regularisation, EMA, previews, saves and an exact resume (finetune_loop)."""
        finetune_loop.train(self)


def main() -> None:
    parser = training_arguments("Finetune the MOSS-Image DiT in maltrainer")
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
        print(MOSSImageFinetuneTrainer.suggest_steps(config, args.suggest_steps))
        return
    if args.resume_from:
        config.setdefault("train", {})["resume_from"] = args.resume_from
    start_run(config)
    trainer = MOSSImageFinetuneTrainer(config, dry_run=args.dry_run, dry_run_step_time=args.dry_run_step_time)
    trainer.config_path = Path(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
