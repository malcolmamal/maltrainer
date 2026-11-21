"""Running the trainer end to end on the tiny world."""

from __future__ import annotations

from typing import Any, cast

import torch
from safetensors.torch import load_file

from train_visage_lora import VisageLoRATrainer

from .world import Workspace


def run_lora(config: dict[str, Any]) -> VisageLoRATrainer:
    trainer = VisageLoRATrainer(config)
    trainer.run()
    return trainer


def prepared(config: dict[str, Any]) -> VisageLoRATrainer:
    """A trainer with its data cached, its model and adapters loaded and its optimizer built, before any step."""
    trainer = VisageLoRATrainer(config)
    trainer.precache_dataset()
    trainer.load_transformer_and_lora()
    trainer.setup_optimizer()
    return trainer


def trained(ws: Workspace, images: int = 2, **sections: Any) -> tuple[VisageLoRATrainer, dict[str, torch.Tensor]]:
    """Run a LoRA on ``images`` images; the trainer and the final file's tensors."""
    ws.add_images(images)
    trainer = run_lora(ws.config(**sections))
    return trainer, load_file(str(ws.out / "visage.safetensors"))


def progress_lines(ws: Workspace) -> list[str]:
    """The ``Step:`` lines of the run's training.log (the progress logger writes only there)."""
    return [line for line in (ws.out / "training.log").read_text(encoding="utf-8").splitlines() if "Step:" in line]


def resolved_optimizer(trainer: VisageLoRATrainer) -> dict[str, Any]:
    """The optimizer settings the trainer recorded in its run metadata."""
    return cast("dict[str, Any]", trainer.run_metadata["optimizer_resolved"])
