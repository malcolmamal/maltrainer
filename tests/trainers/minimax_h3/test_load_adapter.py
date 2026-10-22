"""The frozen training (assistant) adapter attached beside the trainable H3 LoRA."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

import train_minimax_h3_lora as h3
from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRANetwork
from tests.maltrainer.models.minimax_h3.h3_adapters import hand_pair, write_adapter

from .fakes import Fakes
from .runs import cached, linear
from .world import Workspace, same_tensors


def _loaded(ws: Workspace, **sections: Any) -> h3.MiniMaxH3LoRATrainer:
    ws.add_images(1)
    trainer = cached(ws.config(**sections))
    trainer.load_transformer_and_lora()
    return trainer


def _assistant(ws: Workspace) -> Path:
    down, up = hand_pair(16, 16)
    return write_adapter(ws.root / "assistant.safetensors", {"blocks.0.mlp.fc1": (down, up, 2.0)})


@pytest.mark.parametrize("where", ["model", "lora"])
def test_a_training_adapter_is_attached_live_and_is_not_trainable(workspace: Workspace, where: str) -> None:
    path = _assistant(workspace)

    trainer = _loaded(workspace, **{where: {"training_adapter_path": str(path)}})

    assert trainer.run_metadata["training_adapter"] == "assistant.safetensors"
    assert trainer.run_metadata["training_adapter_modules"] == 1
    assert trainer.run_metadata["trainable_parameters"] == 4 * 608


def test_the_training_adapter_changes_what_the_base_predicts_without_touching_its_weights(
    workspace: Workspace, fakes: Fakes
) -> None:
    plain, assisted = _loaded(workspace), None
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(1))
    before = linear(plain, "blocks.0.mlp.fc1")(x)
    workspace.root.joinpath("again").mkdir()
    assisted = _loaded(Workspace(workspace.root / "again"), model={"training_adapter_path": str(_assistant(workspace))})

    after = linear(assisted, "blocks.0.mlp.fc1")(x)

    assert not torch.allclose(before, after)
    assert same_tensors(dict(assisted.transformer.state_dict()), fakes.reference_state())


def test_the_adapter_multiplier_and_adaln_grid_reach_the_loader(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []
    real: Callable[..., int] = MiniMaxH3LoRANetwork.load_assistant_adapter

    def spy(self: MiniMaxH3LoRANetwork, path: str, **options: Any) -> int:
        seen.append({"path": path, **options})
        return real(self, path, **{**options, "adaln_grid_path": None})

    monkeypatch.setattr(MiniMaxH3LoRANetwork, "load_assistant_adapter", spy)

    _loaded(
        workspace,
        model={"training_adapter_path": str(_assistant(workspace)), "training_adapter_adaln_grid": "grid.safetensors"},
        lora={"training_adapter_multiplier": 0.25},
    )

    assert seen[0]["multiplier"] == 0.25 and seen[0]["adaln_grid_path"] == "grid.safetensors"
    assert seen[0]["dtype"] is torch.float32 and seen[0]["device"] == torch.device("cpu")
