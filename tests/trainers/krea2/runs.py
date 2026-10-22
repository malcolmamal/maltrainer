"""Running the trainers end to end on the tiny world, and comparing what two runs left behind."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file

from train_krea2_finetune import Krea2FinetuneTrainer
from train_krea2_lora import Krea2LoRATrainer

from .world import Workspace


def run_lora(config: dict[str, Any]) -> Krea2LoRATrainer:
    trainer = Krea2LoRATrainer(config)
    trainer.run()
    return trainer


def run_finetune(config: dict[str, Any]) -> Krea2FinetuneTrainer:
    trainer = Krea2FinetuneTrainer(config)
    trainer.run()
    return trainer


def prepared_lora(config: dict[str, Any]) -> Krea2LoRATrainer:
    """A trainer with its data cached, its model and adapters loaded and its optimizer built, before any step."""
    trainer = Krea2LoRATrainer(config)
    trainer.precache_dataset()
    trainer.load_transformer_and_lora()
    trainer.setup_optimizer()
    return trainer


def prepared_finetune(config: dict[str, Any]) -> Krea2FinetuneTrainer:
    """A finetune trainer with its data cached, its model loaded and its optimizers built, before any step."""
    trainer = Krea2FinetuneTrainer(config)
    trainer.precache_dataset()
    trainer.load_transformer_and_lora()
    trainer.setup_optimizer()
    return trainer


def trained_lora(ws: Workspace, images: int = 2, **sections: Any) -> tuple[Krea2LoRATrainer, dict[str, torch.Tensor]]:
    """Run a LoRA on ``images`` images; the trainer and the final file's tensors."""
    ws.add_images(images)
    trainer = run_lora(ws.lora_config(**sections))
    return trainer, load_file(str(ws.out / "k2.safetensors"))


def progress_lines(ws: Workspace) -> list[str]:
    """The ``Step:`` lines of the run's training.log (the progress logger writes only there)."""
    return [line for line in (ws.out / "training.log").read_text(encoding="utf-8").splitlines() if "Step:" in line]


def resolved_optimizer(trainer: Krea2LoRATrainer) -> dict[str, Any]:
    """The optimizer settings the trainer recorded in its run metadata."""
    return cast("dict[str, Any]", trainer.run_metadata["optimizer_resolved"])


def leaves(value: Any, path: str = "") -> dict[str, Any]:
    """Every leaf of nested dicts / lists (optimizer states, counters), keyed by its path."""
    if isinstance(value, dict):
        children = ((f"{path}/{key}", item) for key, item in value.items())
    elif isinstance(value, list | tuple):
        children = ((f"{path}[{i}]", item) for i, item in enumerate(value))
    else:
        return {path: value}
    found: dict[str, Any] = {}
    for child_path, item in children:
        found.update(leaves(item, child_path))
    return found


def final_weights(folder: Path, name: str = "ft") -> dict[str, torch.Tensor]:
    """The tensors of a finetune's final checkpoint in ``folder``."""
    return load_file(str(folder / f"{name}.safetensors"))
