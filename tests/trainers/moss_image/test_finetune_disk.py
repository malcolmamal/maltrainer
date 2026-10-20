"""A full disk or a failed write never ends a run early: the save is skipped, cleaned up or waited for."""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import train_moss_image_finetune as finetune

from .runs import run_finetune
from .world import Workspace


def _trained(ws: Workspace, **train: Any) -> finetune.MOSSImageFinetuneTrainer:
    ws.add_images("p", 2)
    return run_finetune(ws.config(train={"max_steps": 1, **train}))


def _free(monkeypatch: pytest.MonkeyPatch, values: Callable[[Path], int]) -> None:
    monkeypatch.setattr(finetune.MOSSImageFinetuneTrainer, "_free_bytes", staticmethod(values))


def test_a_numbered_save_that_does_not_fit_is_skipped_and_reported(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)
    _free(monkeypatch, lambda _folder: 0)

    with caplog.at_level(logging.ERROR):
        saved = trainer.save_checkpoint(7)

    assert saved is False
    assert not (workspace.out / "ft_step_7.safetensors").exists()
    assert "Not saving step 7: it needs" in caplog.text


def test_the_final_save_waits_for_space_in_thirty_second_rounds(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 1)
    sleeps: list[float] = []
    rounds = iter([0, 0, 0, 0, 1 << 60])
    monkeypatch.setattr(time, "sleep", sleeps.append)
    _free(monkeypatch, lambda _folder: next(rounds, 1 << 60))

    with caplog.at_level(logging.ERROR):
        run_finetune(workspace.config(train={"save_state": False}))

    assert sleeps == [30, 30]
    assert (workspace.out / "ft.safetensors").exists()
    assert "The final model needs" in caplog.text


def test_a_state_that_does_not_fit_is_skipped_but_the_model_is_kept(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 1)
    model_size = workspace.layout.stat().st_size
    _free(monkeypatch, lambda _folder: model_size + finetune.MOSSImageFinetuneTrainer.DISK_MARGIN)

    with caplog.at_level(logging.ERROR):
        run_finetune(workspace.config())

    assert (workspace.out / "ft.safetensors").exists() and not (workspace.out / "ft_state").exists()
    assert "Not writing the resume state for step final" in caplog.text


def _failing_save(partial: Path) -> Callable[..., None]:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        Path(str(partial) + ".tmp").write_bytes(b"half")
        raise OSError("disk full")

    return fail


def test_a_failed_numbered_save_removes_its_partial_file_and_training_goes_on(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)
    monkeypatch.setattr(finetune, "save_full_model", _failing_save(workspace.out / "ft_step_3.safetensors"))

    with caplog.at_level(logging.ERROR):
        saved = trainer.save_checkpoint(3)

    assert saved is False
    assert not list(workspace.out.glob("ft_step_3*"))
    assert "Saving step 3 failed (disk full); the partial file is removed and training continues." in caplog.text


def test_a_failed_final_save_removes_its_partial_file_and_raises(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _trained(workspace)
    monkeypatch.setattr(finetune, "save_full_model", _failing_save(workspace.out / "ft_new.safetensors"))
    monkeypatch.setattr(trainer, "_name", lambda: "ft_new")

    with pytest.raises(OSError, match=r"disk full"):
        trainer.save_checkpoint("final")

    assert not list(workspace.out.glob("ft_new*"))


def test_a_failed_state_write_keeps_the_previous_state_and_the_new_model(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)
    marker = workspace.out / "ft_state" / "marker.txt"
    marker.write_text("previous", encoding="utf-8")

    def fail(_state_dir: Path, _weights: Path | None = None) -> None:
        (workspace.out / "ft_state.tmp").mkdir()
        raise OSError("disk full")

    monkeypatch.setattr(trainer, "_save_state", fail)
    with caplog.at_level(logging.ERROR):
        saved = trainer.save_checkpoint(5)

    assert saved is True
    assert (workspace.out / "ft_step_5.safetensors").exists() and marker.read_text(encoding="utf-8") == "previous"
    assert not (workspace.out / "ft_state.tmp").exists()
    assert "Writing the resume state for step 5 failed" in caplog.text


def test_the_state_size_estimate_counts_weights_optimizer_and_ema(workspace: Workspace) -> None:
    trainer = _trained(workspace, ema_decay=0.5)
    assert trainer.ema is not None
    optimizer_bytes = sum(
        t.numel() * t.element_size()
        for state in trainer.optimizer.state.values()
        for t in state.values()
        if hasattr(t, "numel")
    )
    ema_bytes = sum(t.numel() * 4 for t in trainer.ema.shadow)

    assert trainer._state_bytes(hard_linked=True) == optimizer_bytes + ema_bytes + (16 << 20)
    assert trainer._state_bytes(hard_linked=False) == trainer._state_bytes(True) + workspace.layout.stat().st_size


def test_a_dataset_folder_the_cache_cannot_write_to_still_trains(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refused(**_options: object) -> tuple[int, str]:
        raise PermissionError(13, "Permission denied")  # a read-only dataset folder

    monkeypatch.setattr(tempfile, "mkstemp", refused)

    workspace.add_images("p", 2)
    config = workspace.config(dataset={"cache": True}, train={"max_steps": 1})

    with caplog.at_level(logging.WARNING, logger="maltrainer.data.disk_cache"):
        trainer = run_finetune(config)

    assert trainer.global_step == 1
    assert "Not caching latents/" in caplog.text
    assert "Not caching text/" in caplog.text
