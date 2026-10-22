"""The command line: a run from a config file, --resume-from, --dry-run and --suggest-steps."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

import train_moss_image_finetune as finetune
from maltrainer.utils.config import load_config

from .fakes import Fakes
from .world import Workspace


def _command(monkeypatch: pytest.MonkeyPatch, ws: Workspace, config: dict[str, Any], *args: str) -> Path:
    path = ws.root / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_moss_image_finetune.py", "--config", str(path), *args])
    return path


def test_main_trains_from_a_config_file_and_keeps_a_copy_of_the_config(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images("p", 2)
    config = workspace.config()
    _command(monkeypatch, workspace, config)

    finetune.main()

    assert (workspace.out / "ft.safetensors").exists()
    assert load_config(str(workspace.out / "config.yaml")) == config


def test_resume_from_on_the_command_line_overrides_the_config(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 3)
    _command(monkeypatch, workspace, workspace.config(train={"max_steps": 2}))
    finetune.main()
    _command(
        monkeypatch,
        workspace,
        workspace.config(train={"max_steps": 3}),
        "--resume-from",
        str(workspace.out / "ft_state"),
    )

    with caplog.at_level(logging.INFO):
        finetune.main()

    assert load_config(str(workspace.out / "config.yaml"))["train"]["resume_from"] == str(workspace.out / "ft_state")
    assert "Resumed at step 2 (epoch 1, sample 2 of its order)" in caplog.text


def test_a_dry_run_plans_from_the_folders_without_loading_any_model(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("a", 2)
    workspace.add_images("b", 3)
    workspace.add_images(workspace.root / "reg", 4)
    config = workspace.config(
        dataset={"folder_repeats": {"a": 2}, "reg_path": str(workspace.root / "reg"), "reg_repeats": 1},
        train={"max_steps": 0, "num_epochs": 2},
    )
    _command(monkeypatch, workspace, config, "--dry-run", "--dry-run-step-time", "0")

    with caplog.at_level(logging.INFO):
        finetune.main()

    log = (workspace.out / "training.log").read_text(encoding="utf-8")
    assert "Dry run: 11 samples, 11 steps/epoch -> 2 epochs, 22 total steps" in caplog.text
    assert "Step: 22/22 [100%], Loss: 0.000000, Epoch: 2/2, LR: 0.00e+00 (dry run)" in log
    assert "Dry run complete: would save final checkpoint at step 22" in caplog.text
    assert (fakes.vae_loads, fakes.text_loads) == (0, [])
    assert sorted(path.name for path in workspace.out.iterdir()) == ["config.yaml", "training.log"]


def test_suggest_steps_prints_the_epoch_size_and_notes_and_trains_nothing(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.add_images("zed", 2, caption="photo of zed")
    workspace.add_images("ann", 3, caption="photo of ann")
    workspace.add_images(workspace.root / "reg", 4)
    config = workspace.config(
        dataset={"folder_repeats": {"ann": 2, "nobody": 3}, "reg_path": str(workspace.root / "reg"), "reg_repeats": 1},
        prior={"weight": 1.0, "classes": {"zed": "man"}},
        train={"max_steps": 10},
    )
    _command(monkeypatch, workspace, config, "--suggest-steps", "5")

    finetune.main()

    out = capsys.readouterr().out
    assert "Subjects:" in out and "per-folder repeats" in out and "Regularisation:" in out
    assert "One epoch: 12 samples = 12 steps" in out
    assert "Suggested for 5 epochs:  train.max_steps: 60" in out
    assert "Configured now:  max_steps 10 = 0.8 epochs" in out
    assert "Note: dataset.folder_repeats names nobody, which matches no folder (no effect)" in out
    assert "Note: prior.classes has no entry for ann" in out
    assert (fakes.vae_loads, workspace.out.exists()) == (0, False)


def test_suggest_steps_without_a_number_means_the_default_epochs(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.add_images("p", 3)
    _command(monkeypatch, workspace, workspace.config(), "--suggest-steps")

    finetune.main()

    assert "Suggested for 22 epochs:  train.max_steps: 66" in capsys.readouterr().out
