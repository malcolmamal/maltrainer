"""Starting a run: the command line, the dry run, refused configs and a run that fails or is interrupted."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

import train_ltx25_lora as trainer_module
from maltrainer.utils.config import load_config
from train_ltx25_lora import LTX25LoRATrainer

from .fakes import Fakes
from .runs import progress_lines, run_lora
from .world import Workspace


def _command(monkeypatch: pytest.MonkeyPatch, ws: Workspace, config: dict[str, Any], *args: str) -> None:
    path = ws.root / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_ltx25_lora.py", "--config", str(path), *args])


def test_main_trains_from_a_config_file_and_keeps_a_copy_of_the_config(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    config = workspace.config(logging={"level": "debug"})
    _command(monkeypatch, workspace, config)

    trainer_module.main()

    assert (workspace.out / "ltx25.safetensors").exists()
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
    assert "Dry run complete: would save final checkpoint at step 12" in caplog.text
    assert fakes.loads == []
    assert sorted(path.name for path in workspace.out.iterdir()) == ["config.yaml", "training.log"]


def test_a_dry_run_of_a_missing_folder_still_plans_one_sample_and_needs_no_output_section(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(workspace.root)
    config = workspace.config(dataset={"path": str(workspace.root / "nowhere")}, train={"max_steps": 2})
    del config["output"]
    _command(monkeypatch, workspace, config, "--dry-run", "--dry-run-step-time", "0")

    with caplog.at_level(logging.INFO):
        trainer_module.main()

    assert "Dry run: 1 samples, 1 steps/epoch -> 2 epochs, 2 total steps" in caplog.text
    assert (workspace.root / "output" / "training.log").exists() and not workspace.out.exists()


def test_a_batch_size_other_than_one_is_refused_before_anything_starts(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match=r"LTX-2\.5 trains at batch_size 1 .*gradient_accumulation_steps"):
        LTX25LoRATrainer(workspace.config(train={"batch_size": 2}))

    assert not workspace.out.exists()


def test_a_failing_run_logs_the_error_with_its_traceback_and_raises(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)
    workspace.checkpoint.unlink()

    with caplog.at_level(logging.ERROR), pytest.raises(FileNotFoundError, match=r"LTX-2\.5 transformer not found at"):
        run_lora(workspace.config())

    assert "An error occurred during training: LTX-2.5 transformer not found" in caplog.text
    assert caplog.records[-1].exc_info is not None


def test_ctrl_c_ends_the_run_quietly_without_a_final_file(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)

    def interrupted(_self: LTX25LoRATrainer) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(LTX25LoRATrainer, "load_transformer_and_lora", interrupted)

    with caplog.at_level(logging.INFO):
        run_lora(workspace.config())

    assert "Training interrupted by user. Exiting..." in caplog.text
    assert list(Path(workspace.out).glob("*.safetensors")) == []
