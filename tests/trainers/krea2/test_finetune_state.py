"""The resume state a save writes and what resuming does with it: files, counters, mismatches."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from train_krea2_finetune import STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER

from .runs import final_weights, run_finetune
from .world import Workspace, same_values

STATE_FILES = {STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER}


def _state(ws: Workspace) -> Path:
    return ws.out / "ft_state"


def _counters(ws: Workspace) -> dict[str, Any]:
    return torch.load(_state(ws) / STATE_TRAINER, map_location="cpu", weights_only=False)


def _resume(ws: Workspace, **sections: Any) -> Any:
    return run_finetune(
        ws.finetune_config(train={"resume_from": str(_state(ws)), **sections.pop("train", {})}, **sections)
    )


def test_a_save_writes_the_weights_the_optimizers_and_the_counters(workspace: Workspace) -> None:
    workspace.add_images(3)

    run_finetune(workspace.finetune_config(train={"max_steps": 2}))

    assert {path.name for path in _state(workspace).iterdir()} == STATE_FILES
    assert not (workspace.out / "ft_state.tmp").exists() and not (workspace.out / "ft_state.old").exists()
    counters = _counters(workspace)
    assert (counters["global_step"], counters["epoch"], counters["position"]) == (2, 0, 2)
    assert (counters["dataset_samples"], counters["cuda_rng"], len(counters["dataset_fingerprint"])) == (3, None, 40)
    optimizers = torch.load(_state(workspace) / STATE_OPTIMIZER, map_location="cpu", weights_only=False)
    assert optimizers["rotation"]["windows"] == ["attention", "mlp.gate", "mlp.up", "mlp.down"]
    assert list(optimizers["rotation"]["optimizers"]) == [0] and optimizers["always"] is not None


def test_the_state_hard_links_the_checkpoint_it_was_saved_with(workspace: Workspace) -> None:
    workspace.add_images(1)

    run_finetune(workspace.finetune_config())

    assert (_state(workspace) / STATE_MODEL).samefile(workspace.out / "ft.safetensors")


def test_a_failed_hard_link_falls_back_to_a_copy(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)

    def refuse(_source: Any, _target: Any) -> None:
        raise OSError("different volume")

    monkeypatch.setattr(os, "link", refuse)
    with caplog.at_level(logging.INFO):
        run_finetune(workspace.finetune_config())

    assert not (_state(workspace) / STATE_MODEL).samefile(workspace.out / "ft.safetensors")
    assert same_values(final_weights(_state(workspace), "model"), final_weights(workspace.out))
    assert "Could not hard-link the state weights (different volume); copying" in caplog.text


def test_only_the_latest_state_is_kept_and_save_state_false_writes_none(workspace: Workspace) -> None:
    workspace.add_images(2)

    run_finetune(workspace.finetune_config(train={"save_every": 1, "max_steps": 3}))
    assert sorted(path.name for path in workspace.out.iterdir() if path.is_dir()) == ["ft_state"]
    assert _counters(workspace)["global_step"] == 3

    run_finetune(workspace.finetune_config(train={"save_state": False}, output={"path": str(workspace.root / "bare")}))
    assert not (workspace.root / "bare" / "ft_state").exists()
