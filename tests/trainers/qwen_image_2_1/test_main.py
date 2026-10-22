"""The Qwen command line, dry run and error handling."""

from __future__ import annotations

import logging
import sys
from typing import Any

import pytest
import yaml

import train_qwen_image_2_1_lora as qwen
from maltrainer.training import progress

from .fakes import Fakes
from .runs import run_qwen
from .world import Workspace


def _command(monkeypatch: pytest.MonkeyPatch, ws: Workspace, config: dict[str, Any], *args: str) -> None:
    path = ws.root / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_qwen_image_2_1_lora.py", "--config", str(path), *args])


def test_main_trains_from_a_config_file_and_keeps_a_copy_of_it(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    _command(monkeypatch, workspace, workspace.config())

    qwen.main()

    assert (workspace.out / "qwen_image_2_1_lora.safetensors").exists()
    assert yaml.safe_load((workspace.out / "config.yaml").read_text(encoding="utf-8"))["lora"] == {
        "rank": 4,
        "alpha": 4,
    }


def test_a_dry_run_ticks_through_the_planned_steps_without_loading_any_model(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    config = workspace.config(dataset={"repeats": 2}, train={"max_steps": 0, "num_epochs": 2, "save_every": 5})
    _command(monkeypatch, workspace, config, "--dry-run", "--dry-run-step-time", "0")

    with caplog.at_level(logging.INFO):
        qwen.main()

    log = (workspace.out / "training.log").read_text(encoding="utf-8")
    assert "Dry run: 6 samples, 6 steps/epoch -> 2 epochs, 12 total steps (0.000s per simulated step)" in caplog.text
    assert "Step: 7/12 [58%], Loss: 0.000000, Epoch: 2/2, LR: 0.00e+00 (dry run)" in log
    assert [m for m in caplog.messages if m.startswith("Dry run") and "would save" in m] == [
        "Dry run: would save checkpoint at step 5",
        "Dry run: would save checkpoint at step 10",
        "Dry run complete: would save final checkpoint at step 12",
    ]
    assert sum(fakes.loads.values()) == 0
    assert sorted(path.name for path in workspace.out.iterdir()) == ["config.yaml", "training.log"]


def test_a_dry_run_waits_the_step_time_between_steps(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace.add_images(2)
    waits: list[float] = []
    monkeypatch.setattr(progress.time, "sleep", waits.append)
    _command(monkeypatch, workspace, workspace.config(train={"max_steps": 2}), "--dry-run")

    qwen.main()

    assert waits == [0.025, 0.025]


def test_a_dry_run_defaults_to_a_checkpoint_every_five_hundred_steps(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)
    config = workspace.config(train={"max_steps": 501})
    del config["train"]["save_every"]

    with caplog.at_level(logging.INFO):
        qwen.QwenImage21LoRATrainer(config, dry_run=True, dry_run_step_time=0.0).run()

    assert "Dry run: would save checkpoint at step 500" in caplog.text


def test_without_an_output_section_main_writes_no_config_copy_and_logs_to_the_default_folder(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(1)
    config = workspace.config()
    del config["output"]
    monkeypatch.chdir(workspace.root)
    _command(monkeypatch, workspace, config, "--dry-run", "--dry-run-step-time", "0")

    qwen.main()

    assert sorted(p.name for p in (workspace.root / "output").iterdir()) == ["training.log"]


def test_ctrl_c_during_setup_is_logged_and_ends_quietly(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)

    def interrupt(_self: qwen.QwenImage21LoRATrainer) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(qwen.QwenImage21LoRATrainer, "precache_dataset", interrupt)

    with caplog.at_level(logging.INFO):
        run_qwen(workspace.config())

    assert "Training interrupted by user. Exiting..." in caplog.text


def test_any_other_error_is_logged_with_its_traceback_and_raised_again(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)

    def fail(_self: qwen.QwenImage21LoRATrainer) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(qwen.QwenImage21LoRATrainer, "precache_dataset", fail)

    with pytest.raises(RuntimeError, match="boom"):
        run_qwen(workspace.config())

    assert "An error occurred during training: boom" in caplog.text
