"""The finetune's data beyond one folder: per-folder repeats, regularisation images at a smaller step, and
--suggest-steps for both (the shared finetune data, maltrainer/trainers/finetune/dataset.py, on Krea 2)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

import train_krea2_finetune as finetune
from maltrainer.trainers.krea2 import finetune_step
from train_krea2_finetune import Krea2FinetuneTrainer

from .runs import run_finetune
from .world import Workspace, write_image


def _people(ws: Workspace, **counts: int) -> None:
    for name, count in counts.items():
        for i in range(count):
            write_image(ws.data / name / f"{name}{i}.png", f"a photo of {name}")


def _reg(ws: Workspace, count: int) -> Path:
    for i in range(count):
        write_image(ws.root / "reg" / f"r{i}.png", "a photo of a person")
    return ws.root / "reg"


def _cached(config: dict[str, Any]) -> Krea2FinetuneTrainer:
    trainer = Krea2FinetuneTrainer(config)
    trainer.precache_dataset()
    return trainer


def _folder(trainer: Krea2FinetuneTrainer, index: int) -> str:
    return Path(trainer.samples[index][0]).parent.name


def test_folder_repeats_set_how_often_each_folders_images_come_in_an_epoch(workspace: Workspace) -> None:
    _people(workspace, ann=2, ben=2)
    trainer = _cached(workspace.finetune_config(dataset={"folder_repeats": {"ann": 3}}))

    order = trainer._epoch_order()

    assert Counter(_folder(trainer, i) for i in order) == {"ann": 6, "ben": 2}
    assert trainer._epoch_size() == 8


def test_an_epoch_of_folder_repeats_is_the_step_count_of_an_epoch(workspace: Workspace) -> None:
    _people(workspace, ann=2, ben=2)
    trainer = run_finetune(workspace.finetune_config(dataset={"folder_repeats": {"ben": 2.5}}, train={"max_steps": 1}))
    assert trainer.run_metadata["steps_per_epoch"] == 2 + 5  # ben: both twice, and one of the two once more


def test_regularisation_images_join_the_epoch_at_reg_repeats_with_their_own_caption(workspace: Workspace) -> None:
    _people(workspace, ann=2)
    reg = _reg(workspace, 3)
    (reg / "r2.txt").unlink()
    trainer = _cached(
        workspace.finetune_config(dataset={"reg_path": str(reg), "reg_repeats": 2, "reg_default_caption": "a person"})
    )

    assert trainer.reg_keys == {str(reg / f"r{i}.png") for i in range(3)}
    assert dict(trainer.samples)[str(reg / "r2.png")] == "a person"
    assert Counter(_folder(trainer, i) for i in trainer._epoch_order()) == {"ann": 2, "reg": 6}


def test_regularisation_images_step_at_reg_lr_multiplier_on_every_optimizer(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    _people(workspace, ann=2)
    reg = _reg(workspace, 2)
    scales: list[tuple[str, float]] = []
    keys: list[str] = []
    real_step = Krea2FinetuneTrainer._training_step

    def training_step(self: Krea2FinetuneTrainer, index: int, shift_logged: list[bool] | None = None) -> Any:
        keys.append(self.samples[index][0])
        return real_step(self, index, shift_logged)

    def set_step_scale(optimizer: Any, scale: float) -> None:
        scales.append((optimizer.param_groups[0]["name"], scale))
        optimizer.lr_scale = scale

    monkeypatch.setattr(Krea2FinetuneTrainer, "_training_step", training_step)
    monkeypatch.setattr(finetune_step, "set_step_scale", set_step_scale)
    config = workspace.finetune_config(
        dataset={"reg_path": str(reg)}, train={"max_steps": 8, "reg_lr_multiplier": 0.25}
    )

    trainer = run_finetune(config)

    expected = [0.25 if key in trainer.reg_keys else 1.0 for key in keys]
    assert 0.25 in expected and 1.0 in expected
    assert [scale for name, scale in scales if name == "txtfusion"] == expected
    assert [scale for name, scale in scales if name != "txtfusion"] == expected
    assert trainer.run_metadata["reg_images"] == 2 and trainer.run_metadata["reg_lr_multiplier"] == 0.25


def test_without_regularisation_images_the_step_size_is_never_touched(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    calls: list[float] = []
    monkeypatch.setattr(finetune_step, "set_step_scale", lambda _optimizer, scale: calls.append(scale))

    trainer = run_finetune(workspace.finetune_config())

    assert calls == [] and "reg_images" not in trainer.run_metadata


def test_a_regularisation_folder_without_images_is_refused(workspace: Workspace) -> None:
    workspace.add_images(1)
    (workspace.root / "empty").mkdir()
    with pytest.raises(ValueError, match=r"dataset\.reg_path .*empty has no images"):
        _cached(workspace.finetune_config(dataset={"reg_path": str(workspace.root / "empty")}))


def test_a_wrong_base_model_path_fails_before_any_caching(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = Krea2FinetuneTrainer(workspace.finetune_config(model={"transformer_path": str(workspace.root / "no")}))
    with pytest.raises(FileNotFoundError):
        trainer.precache_dataset()
    assert trainer.samples == []


def test_suggest_steps_counts_subjects_and_regularisation_and_trains_nothing(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _people(workspace, ann=3, ben=1)
    reg = _reg(workspace, 4)
    config = workspace.finetune_config(dataset={"reg_path": str(reg), "folder_repeats": {"ben": 3}})
    path = workspace.root / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_krea2_finetune.py", "--config", str(path), "--suggest-steps", "2"])

    finetune.main()

    report = capsys.readouterr().out
    assert "Subjects: " in report and "Regularisation: " in report
    assert "One epoch: 10 samples = 10 steps" in report  # ann 3, ben 1 x 3, reg 4
    assert "Suggested for 2 epochs:  train.max_steps: 20" in report
    assert not workspace.out.exists()
