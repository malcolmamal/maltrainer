"""Muon on the Krea 2 LoRA: the shared optimizer builder reaches it and the run trains."""

from __future__ import annotations

from .runs import resolved_optimizer, trained_lora
from .world import Workspace


def test_a_krea2_lora_trains_with_muon(workspace: Workspace) -> None:
    trainer, weights = trained_lora(workspace, train={"optimizer": "muon", "learning_rate": 1e-4})

    assert resolved_optimizer(trainer)["name"] == "muon"
    assert any(key.endswith("lora_B.weight") and value.abs().max() > 0 for key, value in weights.items())
