"""A full disk or a failed write never ends a run early: the save is skipped, cleaned up or waited for."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from maltrainer.trainers.krea2 import finetune_save
from train_krea2_finetune import Krea2FinetuneTrainer

from .runs import run_finetune
from .world import Workspace


def _trained(ws: Workspace, **train: Any) -> Krea2FinetuneTrainer:
    ws.add_images(2)
    return run_finetune(ws.finetune_config(train={"max_steps": 1, **train}))


def _failing_save(partial: Path) -> Any:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        Path(str(partial) + ".tmp").write_bytes(b"half")
        raise OSError("disk full")

    return fail


def test_the_free_space_needed_is_the_models_bf16_size_plus_a_gigabyte(workspace: Workspace) -> None:
    trainer = _trained(workspace)

    assert trainer._model_bytes() == 2 * sum(t.numel() for t in trainer.transformer.state_dict().values())
    assert finetune_save.DISK_MARGIN == 1 << 30


def test_a_numbered_save_that_does_not_fit_is_skipped_and_reported(
    workspace: Workspace, disk: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)
    disk.free = 0

    with caplog.at_level(logging.ERROR):
        saved = trainer.save_checkpoint(7)

    assert saved is False and not (workspace.out / "ft_step_7.safetensors").exists()
    assert "Not saving step 7: it needs" in caplog.text and "Training continues." in caplog.text


def test_the_final_save_waits_for_space_in_thirty_second_rounds(
    workspace: Workspace, disk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)
    sleeps: list[float] = []

    def free_up(seconds: float) -> None:
        sleeps.append(seconds)
        disk.free = 1 << 40 if len(sleeps) == 2 else 0

    disk.free = 0
    monkeypatch.setattr(time, "sleep", free_up)

    with caplog.at_level(logging.ERROR):
        run_finetune(workspace.finetune_config(train={"save_state": False}))

    assert sleeps == [30, 30] and (workspace.out / "ft.safetensors").exists()
    assert caplog.text.count("The final model needs") == 2


def test_a_failed_numbered_save_removes_its_partial_file_and_training_goes_on(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)
    monkeypatch.setattr(finetune_save, "save_krea2_model", _failing_save(workspace.out / "ft_step_3.safetensors"))

    with caplog.at_level(logging.ERROR):
        saved = trainer.save_checkpoint(3)

    assert saved is False and not list(workspace.out.glob("ft_step_3*"))
    assert "Saving step 3 failed (disk full); the partial file is removed and training continues." in caplog.text


def test_a_failed_final_save_removes_its_partial_file_and_raises(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _trained(workspace)
    monkeypatch.setattr(trainer, "_name", lambda: "ft_new")
    monkeypatch.setattr(finetune_save, "save_krea2_model", _failing_save(workspace.out / "ft_new.safetensors"))

    with pytest.raises(OSError, match=r"disk full"):
        trainer.save_checkpoint("final")

    assert not list(workspace.out.glob("ft_new*"))


def test_a_failed_state_write_keeps_the_previous_state_and_the_new_model(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)
    marker = workspace.out / "ft_state" / "marker.txt"
    marker.write_text("previous", encoding="utf-8")

    def fail(_state_dir: Path, _weights: Path) -> None:
        (workspace.out / "ft_state.tmp").mkdir()
        raise OSError("disk full")

    monkeypatch.setattr(trainer, "_save_state", fail)
    with caplog.at_level(logging.ERROR):
        saved = trainer.save_checkpoint(5)

    assert saved is True and (workspace.out / "ft_step_5.safetensors").exists()
    assert marker.read_text(encoding="utf-8") == "previous" and not (workspace.out / "ft_state.tmp").exists()
    assert "Writing the resume state for step 5 failed; the previous state is kept." in caplog.text


@pytest.mark.parametrize(
    ("name", "expected"), [("mine", "mine"), ("mine.safetensors", "mine"), (None, "krea2_finetune")]
)
def test_the_checkpoint_name_is_output_name_without_its_extension_or_a_default(
    workspace: Workspace, name: str | None, expected: str
) -> None:
    workspace.add_images(1)
    trainer = Krea2FinetuneTrainer(workspace.finetune_config(output={"name": name}))

    assert trainer._name() == expected


def test_a_numbered_save_is_skipped_when_the_model_fits_but_its_resume_state_does_not(
    workspace: Workspace, disk: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)
    state = finetune_save.state_bytes(trainer)
    disk.free = trainer._model_bytes() + finetune_save.DISK_MARGIN  # the model alone would fit

    with caplog.at_level(logging.ERROR):
        saved = trainer.save_checkpoint(7)

    assert state > 0 and saved is False and not list(workspace.out.glob("ft_step_7*"))
    assert "Not saving step 7" in caplog.text
    disk.free += state
    assert trainer.save_checkpoint(8) is True and (workspace.out / "ft_state").is_dir()


def test_without_a_resume_state_only_the_model_has_to_fit(workspace: Workspace, disk: SimpleNamespace) -> None:
    trainer = _trained(workspace, save_state=False)
    disk.free = trainer._model_bytes() + finetune_save.DISK_MARGIN

    assert trainer.save_checkpoint(7) is True


def test_a_failed_state_write_has_closed_its_file_so_the_unfinished_state_can_be_removed(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On Windows a file still open cannot be deleted: torch.save given a path kept it open while the error
    # was handled, and the 11 GB unfinished state stayed on a full disk
    trainer = _trained(workspace)
    targets: list[Any] = []

    def fail(_obj: Any, target: Any) -> None:
        targets.append(target)
        raise RuntimeError("unexpected pos")

    monkeypatch.setattr(finetune_save.torch, "save", fail)
    assert trainer.save_checkpoint(5) is True

    assert len(targets) == 1 and targets[0].closed
    assert not (workspace.out / "ft_state.tmp").exists()


def test_an_unfinished_state_that_cannot_be_removed_is_reported_with_its_path(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trained(workspace)

    def fail(_state_dir: Path, _weights: Path) -> None:
        (workspace.out / "ft_state.tmp").mkdir()
        raise OSError("disk full")

    monkeypatch.setattr(trainer, "_save_state", fail)
    monkeypatch.setattr(finetune_save.shutil, "rmtree", lambda *_args, **_kwargs: None)  # Windows refuses
    with caplog.at_level(logging.ERROR):
        trainer.save_checkpoint(5)

    assert f"Could not remove {workspace.out / 'ft_state.tmp'}" in caplog.text
