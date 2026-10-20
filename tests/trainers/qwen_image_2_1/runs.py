"""Running the Qwen-Image-2.1 trainer end to end on the tiny world, and inspecting what it left behind."""

from __future__ import annotations

from typing import Any

from accelerate.state import AcceleratorState, GradientState

import train_qwen_image_2_1_lora as qwen
from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRALayer

from .world import Workspace


def run_qwen(config: dict[str, Any]) -> qwen.QwenImage21LoRATrainer:
    trainer = qwen.QwenImage21LoRATrainer(config)
    trainer.run()
    return trainer


def cached(config: dict[str, Any]) -> qwen.QwenImage21LoRATrainer:
    """A trainer that has made its dataset and cached its latents and prompts (nothing else)."""
    trainer = qwen.QwenImage21LoRATrainer(config)
    trainer.create_dataset()
    trainer.precache_dataset()
    return trainer


def loaded(config: dict[str, Any]) -> qwen.QwenImage21LoRATrainer:
    """A trainer whose transformer and adapters are loaded, before the optimizer."""
    trainer = cached(config)
    trainer.load_transformer_and_lora()
    return trainer


def prepared(config: dict[str, Any]) -> qwen.QwenImage21LoRATrainer:
    """A trainer with everything set up and its optimizer built, before any step."""
    trainer = loaded(config)
    trainer.setup_optimizer()
    return trainer


def progress_lines(ws: Workspace) -> list[str]:
    """The ``Step:`` lines of the run's training.log (the progress logger writes only there)."""
    return [line for line in (ws.out / "training.log").read_text(encoding="utf-8").splitlines() if "Step:" in line]


def rerun(config: dict[str, Any]) -> qwen.QwenImage21LoRATrainer:
    """``run_qwen`` after dropping the accelerate singletons, for a second run inside one test."""
    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    return run_qwen(config)


def lora_layer(trainer: qwen.QwenImage21LoRATrainer, key: str) -> QwenImage21LoRALayer:
    layer = trainer.lora_network.lora_layers[key]
    assert isinstance(layer, QwenImage21LoRALayer)
    return layer
