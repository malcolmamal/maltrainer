"""The LoRA trainer's command line, dry run, step suggestions, error handling and checkpoint names."""

from __future__ import annotations

import logging
import sys
from typing import Any

import pytest
import yaml

import train_moss_image_lora as lora

from .fakes import Fakes
from .runs import prepared_lora, run_lora
from .world import Workspace


def _command(monkeypatch: pytest.MonkeyPatch, ws: Workspace, config: dict[str, Any], *args: str) -> None:
    path = ws.root / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_moss_image_lora.py", "--config", str(path), *args])


def test_main_trains_from_a_config_file_and_watches_it(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace.add_images("p", 2)
    _command(monkeypatch, workspace, workspace.lora_config())

    lora.main()

    assert (workspace.out / "moss_image_lora.safetensors").exists() and (workspace.out / "config.yaml").exists()


def test_a_dry_run_ticks_through_the_planned_steps_without_loading_any_model(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("a", 2)
    workspace.add_images("b", 1)
    config = workspace.lora_config(dataset={"repeats": 2}, train={"max_steps": 0, "num_epochs": 2})
    _command(monkeypatch, workspace, config, "--dry-run", "--dry-run-step-time", "0")

    with caplog.at_level(logging.INFO):
        lora.main()

    log = (workspace.out / "training.log").read_text(encoding="utf-8")
    assert "Dry run: 6 samples, 6 steps/epoch -> 2 epochs, 12 total steps" in caplog.text
    assert "Step: 7/12 [58%], Loss: 0.000000, Epoch: 2/2, LR: 0.00e+00 (dry run)" in log
    assert (fakes.vae_loads, fakes.text_loads) == (0, [])
    assert sorted(path.name for path in workspace.out.iterdir()) == ["config.yaml", "training.log"]


def test_a_dry_run_counts_at_least_one_sample_for_an_empty_folder(workspace: Workspace) -> None:
    workspace.data.mkdir()

    trainer = lora.MOSSImageLoRATrainer(workspace.lora_config(dataset={"repeats": 3}))

    assert trainer._dry_run_samples() == 3


def test_suggest_steps_reports_the_epoch_size_for_the_wanted_epochs(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace.add_images("a", 2)
    workspace.add_images("b", 3)
    config = workspace.lora_config(dataset={"repeats": 2}, train={"gradient_accumulation_steps": 2})
    _command(monkeypatch, workspace, config, "--suggest-steps", "4")

    lora.main()

    out = capsys.readouterr().out
    assert "Subjects: " in out and "(5 images, 2 repeats = 10 samples per epoch)" in out
    assert "One epoch: 10 samples = 5 steps (gradient_accumulation_steps 2)" in out
    assert "Suggested for 4 epochs:  train.max_steps: 20" in out
    assert not workspace.out.exists()


def test_the_lora_trainer_has_no_step_notes() -> None:
    assert lora.MOSSImageLoRATrainer.step_notes({"dataset": {}}) == []


def test_ctrl_c_during_setup_is_logged_and_ends_quietly(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 1)

    def interrupt(_self: lora.MOSSImageLoRATrainer) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(lora.MOSSImageLoRATrainer, "precache_dataset", interrupt)

    with caplog.at_level(logging.INFO):
        run_lora(workspace.lora_config())

    assert "Training interrupted by user. Exiting..." in caplog.text


def test_any_other_error_is_logged_with_its_traceback_and_raised_again(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 1)

    def fail(_self: lora.MOSSImageLoRATrainer) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(lora.MOSSImageLoRATrainer, "precache_dataset", fail)

    with pytest.raises(RuntimeError, match="boom"):
        run_lora(workspace.lora_config())

    assert "An error occurred during training: boom" in caplog.text


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"name": "mine"}, "mine.safetensors"),
        ({"name": "mine.safetensors"}, "mine.safetensors"),
        ({"lora_name": "legacy"}, "legacy.safetensors"),
        ({}, "moss_image_lora.safetensors"),
    ],
)
def test_the_checkpoint_name_comes_from_name_then_lora_name_then_the_default(
    workspace: Workspace, output: dict[str, str], expected: str
) -> None:
    workspace.add_images("p", 1)
    config = workspace.lora_config()
    config["output"].update(output)

    run_lora(config)

    assert sorted(path.name for path in workspace.out.glob("*.safetensors")) == [expected]


def test_a_second_process_saves_nothing(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    trainer = prepared_lora(workspace.lora_config())
    trainer.is_main_process = False

    assert trainer.save_checkpoint("final") is None
    assert not list(workspace.out.glob("*.safetensors")) and trainer.last_saved_path is None
