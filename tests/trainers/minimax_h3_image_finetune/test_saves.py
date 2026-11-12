"""Numbered saves, the resume state's contents, and a full disk or failed write never ending a run early."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from maltrainer.trainers.krea2 import finetune_save as shared
from maltrainer.trainers.minimax_h3_image import finetune_save
from tests.trainers.minimax_h3.world import Workspace

from .world import NAME, finetune_config, run_finetune


def _trained(ws: Workspace, **train: Any) -> Any:
    ws.add_images(2)
    return run_finetune(finetune_config(ws, output={"name": NAME}, train={"max_steps": 2, **train}))


def _disk(monkeypatch: pytest.MonkeyPatch, free: list[int]) -> None:
    monkeypatch.setattr(shared.shutil, "disk_usage", lambda _path: SimpleNamespace(free=free[0]))


def test_a_numbered_save_per_save_every_steps_and_only_the_latest_state(workspace: Workspace) -> None:
    _trained(workspace, save_every=1, max_steps=3)

    names = sorted(p.name for p in workspace.out.glob("*.safetensors"))
    assert names == [f"{NAME}.safetensors", f"{NAME}_step_1.safetensors", f"{NAME}_step_2.safetensors"]
    assert sorted(p.name for p in workspace.out.iterdir() if p.is_dir()) == [f"{NAME}_state"]
    counters = torch.load(workspace.out / f"{NAME}_state" / "trainer_state.pt", weights_only=False)
    assert counters["global_step"] == 3 and counters["dataset_samples"] == 2


def test_save_state_false_writes_the_model_only(workspace: Workspace) -> None:
    _trained(workspace, save_state=False)

    assert (workspace.out / f"{NAME}.safetensors").exists() and not (workspace.out / f"{NAME}_state").exists()


def test_the_name_may_carry_the_extension(workspace: Workspace) -> None:
    workspace.add_images(1)

    run_finetune(finetune_config(workspace, output={"name": "mine.safetensors"}, train={"max_steps": 1}))

    assert (workspace.out / "mine.safetensors").exists()


def test_a_numbered_save_that_does_not_fit_is_skipped_and_reported(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)
    _disk(monkeypatch, [0])

    with caplog.at_level(logging.ERROR):
        saved = finetune_save.save_checkpoint(trainer, 7)

    assert saved is False and not (workspace.out / f"{NAME}_step_7.safetensors").exists()
    assert "Not saving step 7: it needs" in caplog.text


def test_a_failed_numbered_write_removes_the_partial_file_and_training_goes_on(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)

    def fail(_source: Any, _rotator: Any, _always: Any, path: Path, _metadata: Any) -> None:
        Path(str(path) + ".tmp").write_bytes(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(finetune_save, "write_int8_model", fail)
    with caplog.at_level(logging.ERROR):
        assert finetune_save.save_checkpoint(trainer, 5) is False

    assert not list(workspace.out.glob("*step_5*")) and "Saving step 5 failed (disk full)" in caplog.text
    with pytest.raises(OSError, match="disk full"):
        finetune_save.save_checkpoint(trainer, "final")


def test_a_failed_state_write_keeps_the_previous_state_and_the_model_is_still_saved(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)
    before = sorted(p.name for p in (workspace.out / f"{NAME}_state").iterdir())

    def fail(_rotator: Any, _always: Any, _path: Path) -> None:
        raise OSError("no space")

    monkeypatch.setattr(finetune_save, "write_resume_weights", fail)
    with caplog.at_level(logging.ERROR):
        assert finetune_save.save_checkpoint(trainer, 9) is True

    assert (workspace.out / f"{NAME}_step_9.safetensors").exists()
    assert sorted(p.name for p in (workspace.out / f"{NAME}_state").iterdir()) == before
    assert "Writing the resume state for step 9 failed" in caplog.text
    assert not (workspace.out / f"{NAME}_state.tmp").exists()


def test_the_final_save_waits_for_space_in_thirty_second_rounds(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)
    free, sleeps = [0], []
    _disk(monkeypatch, free)

    def wait(seconds: float) -> None:
        sleeps.append(seconds)
        free[0] = 1 << 60

    monkeypatch.setattr(time, "sleep", wait)
    with caplog.at_level(logging.ERROR):
        assert finetune_save.save_checkpoint(trainer, "final") is True

    assert sleeps == [30] and "The final model needs" in caplog.text
