"""A resume state whose write fails leaves nothing behind, or says where it is (MOSS-Image finetune)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import train_moss_image_finetune as finetune
from maltrainer.trainers.moss_image import finetune_state
from maltrainer.utils import fsutil

from .runs import run_finetune
from .world import Workspace


def _trained(ws: Workspace) -> finetune.MOSSImageFinetuneTrainer:
    ws.add_images("p", 2)
    return run_finetune(ws.config(train={"max_steps": 1}))


def test_a_failed_state_write_has_closed_its_file_so_the_unfinished_state_can_be_removed(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On Windows a file still open cannot be deleted: torch.save given a path can keep it open while the
    # error is handled, and the unfinished state stays on a full disk
    trainer = _trained(workspace)
    targets: list[Any] = []

    def fail(_obj: Any, target: Any) -> None:
        targets.append(target)
        raise RuntimeError("unexpected pos")

    monkeypatch.setattr(finetune_state.torch, "save", fail)
    assert trainer.save_checkpoint(5) is True

    assert len(targets) == 1 and targets[0].closed
    assert not (workspace.out / "ft_state.tmp").exists()


def test_an_unfinished_state_that_cannot_be_removed_is_reported_with_its_path(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)

    def fail(_state_dir: Path, **_kwargs: Any) -> None:  # called with weights=
        (workspace.out / "ft_state.tmp").mkdir()
        raise OSError("disk full")

    monkeypatch.setattr(trainer, "_save_state", fail)
    monkeypatch.setattr(fsutil.shutil, "rmtree", lambda *_args, **_kwargs: None)  # Windows refuses
    with caplog.at_level(logging.ERROR):
        trainer.save_checkpoint(5)

    assert f"Could not remove {workspace.out / 'ft_state.tmp'}" in caplog.text
