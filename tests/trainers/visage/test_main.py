"""Starting a run: the command line, the dry run, a refused batch size and a run that fails."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

import train_visage_lora as trainer_module
from maltrainer.utils.config import load_config
from train_visage_lora import VisageLoRATrainer

from .fakes import Fakes
from .runs import progress_lines
from .world import Workspace


def _command(monkeypatch: pytest.MonkeyPatch, ws: Workspace, config: dict[str, Any], *args: str) -> Path:
    path = ws.root / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_visage_lora.py", "--config", str(path), *args])
    return path


def test_main_trains_from_a_config_file_and_keeps_a_copy_of_the_config(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    config = workspace.config(logging={"level": "debug"})
    _command(monkeypatch, workspace, config)

    trainer_module.main()

    assert (workspace.out / "visage.safetensors").exists()
    assert load_config(str(workspace.out / "config.yaml")) == config


def test_a_dry_run_plans_from_the_folder_without_loading_any_model(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    _command(monkeypatch, workspace, workspace.config(dataset={"repeats": 2}, train={"max_steps": 0, "num_epochs": 2}))
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--dry-run", "--dry-run-step-time", "0"])

    with caplog.at_level(logging.INFO):
        trainer_module.main()

    assert "Dry run: 6 samples, 6 steps/epoch -> 2 epochs, 12 total steps" in caplog.text
    assert "Step: 12/12 [100%], Loss: 0.000000, Epoch: 2/2, LR: 0.00e+00 (dry run)" in progress_lines(workspace)[-1]
    assert (fakes.vae_loads, fakes.text_loads) == (0, [])


def test_a_batch_size_above_one_is_refused_with_the_accumulation_hint(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match=r"batch_size 1.*gradient_accumulation_steps"):
        VisageLoRATrainer(workspace.config(train={"batch_size": 4}))


def test_the_trainer_is_labelled_for_the_logs_and_the_metadata(workspace: Workspace) -> None:
    trainer = VisageLoRATrainer(workspace.config())

    assert (trainer.label, trainer.default_name, trainer.run_metadata["model_type"]) == (
        "Visage-Image",
        "visage_lora",
        "visage",
    )


def test_a_failing_run_is_logged_and_raised(workspace: Workspace, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="No images found"):
        VisageLoRATrainer(workspace.config()).run()

    assert "An error occurred during training: No images found" in caplog.text


def test_an_interrupt_ends_the_run_quietly(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def interrupt(_trainer: VisageLoRATrainer) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(VisageLoRATrainer, "precache_dataset", interrupt)

    with caplog.at_level(logging.INFO):
        VisageLoRATrainer(workspace.config()).run()

    assert "Training interrupted by user" in caplog.text
