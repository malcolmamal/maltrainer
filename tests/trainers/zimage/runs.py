"""Running a trainer on the tiny world and watching what it computes."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import train_zimage_embedding as embedding
import train_zimage_lora as lora


def watch_losses(trainer: Any, monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Note the loss of every backward pass of ``trainer``; the list fills as it trains."""
    seen: list[float] = []
    real = trainer.accelerator.backward

    def backward(loss: torch.Tensor, **kwargs: Any) -> None:
        seen.append(loss.item())
        real(loss, **kwargs)

    monkeypatch.setattr(trainer.accelerator, "backward", backward)
    return seen


def watch_noise(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    """Keep every ``torch.randn_like`` result: the first of a step is the trainer's noise."""
    seen: list[torch.Tensor] = []
    real = torch.randn_like

    def randn_like(tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        seen.append(real(tensor, **kwargs))
        return seen[-1]

    monkeypatch.setattr(torch, "randn_like", randn_like)
    return seen


def run_lora(config: dict[str, Any]) -> lora.ZImageLoRATrainer:
    trainer = lora.ZImageLoRATrainer(config)
    trainer.run()
    return trainer


def run_embedding(config: dict[str, Any]) -> embedding.ZImageEmbeddingTrainer:
    trainer = embedding.ZImageEmbeddingTrainer(config)
    trainer.run()
    return trainer


def prepared_lora(config: dict[str, Any]) -> lora.ZImageLoRATrainer:
    """A LoRA trainer with its model, network and dataset loaded and its optimizer built, before any step."""
    trainer = lora.ZImageLoRATrainer(config)
    trainer.load_model()
    trainer.create_lora_network()
    trainer.create_dataset()
    trainer.setup_optimizer()
    return trainer


def prepared_embedding(config: dict[str, Any]) -> embedding.ZImageEmbeddingTrainer:
    trainer = embedding.ZImageEmbeddingTrainer(config)
    trainer.load_model()
    trainer.create_embedding()
    trainer.create_dataset()
    trainer.setup_optimizer()
    return trainer


def first_step(ws: Any, fakes: Any, monkeypatch: pytest.MonkeyPatch, **sections: Any) -> tuple[float, float]:
    """Train one image for one step: (that step's loss, the timestep it drew). ``fakes`` forgets older calls."""
    ws.add_images(1)
    fakes.calls.clear()
    trainer = lora.ZImageLoRATrainer(ws.lora_config(**sections))
    losses = watch_losses(trainer, monkeypatch)
    trainer.run()
    return losses[0], 1.0 - fakes.calls[0].t.item()
