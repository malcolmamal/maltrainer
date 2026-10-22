"""The resume state a save writes: its files, the hard link to the checkpoint, what the counters hold."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open

from .runs import run_finetune
from .world import Workspace

STATE_FILES = {"model.safetensors", "optimizer.pt", "trainer_state.pt"}


def _state(ws: Workspace) -> Any:
    return ws.out / "ft_state"


def _counters(ws: Workspace) -> dict[str, Any]:
    return torch.load(_state(ws) / "trainer_state.pt", map_location="cpu", weights_only=False)


def test_a_save_writes_the_weights_the_optimizer_and_the_counters(workspace: Workspace) -> None:
    workspace.add_images("p", 3)

    run_finetune(workspace.config(train={"max_steps": 2}))

    assert {path.name for path in _state(workspace).iterdir()} == STATE_FILES
    assert not (workspace.out / "ft_state.tmp").exists() and not (workspace.out / "ft_state.old").exists()
    counters = _counters(workspace)
    assert (counters["global_step"], counters["epoch"], counters["position"]) == (2, 0, 2)
    assert (counters["group_names"], counters["dataset_samples"], counters["ema_counters"]) == (
        ["attn", "mlp"],
        3,
        None,
    )
    assert len(counters["dataset_fingerprint"]) == 40
    assert counters["cuda_rng"] is None


def test_without_ema_the_state_hard_links_the_checkpoint_it_was_saved_with(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    run_finetune(workspace.config())

    assert (_state(workspace) / "model.safetensors").samefile(workspace.out / "ft.safetensors")


@pytest.mark.parametrize(
    "sections", [{"train": {"ema_decay": 0.5}}, {"output": {"dtype": "float32"}}], ids=["ema", "output_dtype"]
)
def test_a_checkpoint_that_differs_from_the_raw_weights_gets_its_own_copy_in_the_state(
    workspace: Workspace, sections: dict[str, Any]
) -> None:
    workspace.add_images("p", 1)

    run_finetune(workspace.config(**sections))

    copy = _state(workspace) / "model.safetensors"
    assert not copy.samefile(workspace.out / "ft.safetensors")
    with safe_open(str(copy), framework="pt") as handle:
        assert handle.metadata() == {"maltrainer_state_step": "3"}


def test_the_state_holds_the_ema_and_its_counters(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    run_finetune(workspace.config(train={"ema_decay": 0.5, "ema_every": 2}))

    assert (_state(workspace) / "ema.safetensors").exists()
    assert _counters(workspace)["ema_counters"] == {"updates": 1, "calls": 3}


def test_a_failed_hard_link_falls_back_to_writing_copies(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace.add_images("p", 1, caption="a photo of zed")

    def refuse(_source: Any, _target: Any) -> None:
        raise OSError("different volume")

    monkeypatch.setattr(os, "link", refuse)
    run_finetune(workspace.config(prior={"weight": 1.0, "samples_per_image": 1}))

    assert {path.name for path in _state(workspace).iterdir()} == STATE_FILES | {"prior.safetensors"}
    assert not (_state(workspace) / "model.safetensors").samefile(workspace.out / "ft.safetensors")


def test_only_the_latest_state_is_kept(workspace: Workspace) -> None:
    workspace.add_images("p", 2)

    run_finetune(workspace.config(train={"save_every": 1, "max_steps": 3}))

    assert sorted(path.name for path in workspace.out.iterdir() if path.is_dir()) == ["ft_state"]
    assert _counters(workspace)["global_step"] == 3


def test_save_state_false_writes_the_model_alone(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    run_finetune(workspace.config(train={"save_state": False}))

    assert (workspace.out / "ft.safetensors").exists() and not _state(workspace).exists()


def test_a_refused_rename_of_the_finished_state_folder_is_retried_not_lost(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images("p", 3)
    real = Path.rename
    refused: list[Path] = []

    def refuse_once(self: Path, target: Path) -> Path:
        if self.name.endswith("_state.tmp") and not refused:
            refused.append(self)
            raise PermissionError(5, "Access is denied")  # a virus scanner holding the new folder
        return real(self, target)

    monkeypatch.setattr(Path, "rename", refuse_once)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    run_finetune(workspace.config(train={"max_steps": 2}))

    assert len(refused) == 1
    assert {path.name for path in _state(workspace).iterdir()} == STATE_FILES
    assert _counters(workspace)["global_step"] == 2
