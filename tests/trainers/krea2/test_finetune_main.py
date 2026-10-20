"""The finetune's command line: a run from a config file, --resume-from, --dry-run and a run that fails."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

import train_krea2_finetune as finetune
from maltrainer.utils.config import load_config

from .fakes import Fakes
from .runs import progress_lines
from .world import Workspace


def _command(monkeypatch: pytest.MonkeyPatch, ws: Workspace, config: dict[str, Any], *args: str) -> Path:
    path = ws.root / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_krea2_finetune.py", "--config", str(path), *args])
    return path


def test_main_trains_from_a_config_file_and_keeps_a_copy_of_the_config(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    config = workspace.finetune_config()
    _command(monkeypatch, workspace, config)

    finetune.main()

    assert (workspace.out / "ft.safetensors").exists() and (workspace.out / "ft_state").is_dir()
    assert load_config(str(workspace.out / "config.yaml")) == config


def test_resume_from_on_the_command_line_overrides_the_config(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    _command(monkeypatch, workspace, workspace.finetune_config(train={"max_steps": 2}))
    finetune.main()
    state = str(workspace.out / "ft_state")
    _command(monkeypatch, workspace, workspace.finetune_config(train={"max_steps": 3}), "--resume-from", state)

    with caplog.at_level(logging.INFO):
        finetune.main()

    assert load_config(str(workspace.out / "config.yaml"))["train"]["resume_from"] == state
    assert "Resumed at step 2 (epoch 1, sample 2 of its order)" in caplog.text
    assert "Step: 3/3 [100%]" in progress_lines(workspace)[-1]


def test_a_dry_run_plans_from_the_folder_without_loading_any_model(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    config = workspace.finetune_config(dataset={"repeats": 2}, train={"max_steps": 0, "num_epochs": 2})
    _command(monkeypatch, workspace, config, "--dry-run", "--dry-run-step-time", "0")

    with caplog.at_level(logging.INFO):
        finetune.main()

    assert "Dry run: 6 samples, 6 steps/epoch -> 2 epochs, 12 total steps" in caplog.text
    assert "Step: 12/12 [100%], Loss: 0.000000, Epoch: 2/2, LR: 0.00e+00 (dry run)" in progress_lines(workspace)[-1]
    assert (fakes.vae_loads, fakes.text_loads) == (0, [])
    assert sorted(path.name for path in workspace.out.iterdir()) == ["config.yaml", "training.log"]


def test_a_missing_resume_state_fails_the_run_after_logging_it(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)
    _command(monkeypatch, workspace, workspace.finetune_config(), "--resume-from", str(workspace.root / "nowhere"))

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(FileNotFoundError, match=r"train\.resume_from: .*model\.safetensors is missing"),
    ):
        finetune.main()

    assert "An error occurred during training: train.resume_from:" in caplog.text


def test_a_dry_run_needs_no_output_section_and_logs_in_the_working_directory(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(workspace.root)
    config = workspace.finetune_config(dataset={"path": str(workspace.root / "nowhere")}, train={"max_steps": 2})
    del config["output"]
    _command(monkeypatch, workspace, config, "--dry-run", "--dry-run-step-time", "0")

    with caplog.at_level(logging.INFO):
        finetune.main()

    assert "Dry run: 1 samples, 1 steps/epoch -> 2 epochs, 2 total steps" in caplog.text
    assert (workspace.root / "output" / "training.log").exists() and not workspace.out.exists()
