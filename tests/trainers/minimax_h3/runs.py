"""Running the H3 trainer end to end on the tiny world, and inspecting what it left behind."""

from __future__ import annotations

from typing import Any

import torch

import train_minimax_h3_lora as h3
from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRALayer

from .world import Workspace


def run_h3(config: dict[str, Any]) -> h3.MiniMaxH3LoRATrainer:
    trainer = h3.MiniMaxH3LoRATrainer(config)
    trainer.run()
    return trainer


def cached(config: dict[str, Any]) -> h3.MiniMaxH3LoRATrainer:
    """A trainer that has scanned and cached its dataset (nothing else)."""
    trainer = h3.MiniMaxH3LoRATrainer(config)
    trainer.precache_dataset()
    return trainer


def prepared(config: dict[str, Any]) -> h3.MiniMaxH3LoRATrainer:
    """A trainer with its data cached, its model and adapters loaded and its optimizer built, before any step."""
    trainer = cached(config)
    trainer.load_transformer_and_lora()
    trainer.setup_optimizer()
    return trainer


def progress_lines(ws: Workspace) -> list[str]:
    """The ``Step:`` lines of the run's training.log (the progress logger writes only there)."""
    return [line for line in (ws.out / "training.log").read_text(encoding="utf-8").splitlines() if "Step:" in line]


def linear(trainer: h3.MiniMaxH3LoRATrainer, name: str) -> torch.nn.Linear:
    """The transformer's ``torch.nn.Linear`` called ``name`` (its forward may carry adapters)."""
    module = trainer.transformer.get_submodule(name)
    assert isinstance(module, torch.nn.Linear)
    return module


def lora_layer(trainer: h3.MiniMaxH3LoRATrainer, key: str) -> MiniMaxH3LoRALayer:
    layer = trainer.lora_network.lora_layers[key]
    assert isinstance(layer, MiniMaxH3LoRALayer)
    return layer
