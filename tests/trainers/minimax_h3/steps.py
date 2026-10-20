"""Helpers for the single-step tests: a trainer holding one item of each kind, and a recorder of its forwards."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import train_minimax_h3_lora as h3

from .runs import prepared
from .world import Workspace

SHIFT = 2.0


class Calls:
    """The arguments of every ``_forward`` the trainer makes, recorded around the real call."""

    def __init__(self, trainer: h3.MiniMaxH3LoRATrainer, monkeypatch: pytest.MonkeyPatch) -> None:
        self.args: list[tuple[Any, ...]] = []
        original = trainer._forward

        def spy(*args: Any) -> Any:
            self.args.append(args)
            return original(*args)

        monkeypatch.setattr(trainer, "_forward", spy)


def make_trainer(ws: Workspace, **train: Any) -> h3.MiniMaxH3LoRATrainer:
    ws.add_images(1)
    ws.add_clip(seconds=0.3)
    ws.add_voice(caption="she hums softly now")
    return prepared(ws.config(train=train))


def sample_of(trainer: h3.MiniMaxH3LoRATrainer, kind: str) -> h3.H3Sample:
    return next(s for s in trainer.samples if s.kind == kind)


def step(trainer: h3.MiniMaxH3LoRATrainer, kind: str, seed: int = 4) -> tuple[torch.Tensor, dict[str, float]]:
    torch.manual_seed(seed)
    return trainer._training_step(sample_of(trainer, kind), SHIFT)
