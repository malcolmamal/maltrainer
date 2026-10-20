"""The LoRA trainer's run(): its stages, gradient accumulation, interruption and failure handling."""

from __future__ import annotations

import logging
import re
from typing import Any

import pytest

import train_zimage_lora as lora
from maltrainer.trainers.zimage import pipeline

from .runs import run_lora, watch_losses
from .world import Workspace

PIPELINE = pipeline.__name__  # where the run order logs its lines

STAGES = ["load_model", "create_lora_network", "create_dataset", "setup_optimizer", "train"]


def _log_lines(ws: Workspace) -> list[str]:
    text = (ws.out / "training.log").read_text(encoding="utf-8")
    return [line.split(" - progress - INFO - ")[1].split(", Loss")[0] for line in text.splitlines() if "Step:" in line]


def test_the_stages_run_in_order_between_seeding_and_the_summary(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[Any] = []
    trainer = lora.ZImageLoRATrainer(workspace.lora_config(seed=11))
    monkeypatch.setattr(pipeline, "set_seed", lambda seed: seen.append(("seed", seed)))
    for stage in STAGES:
        monkeypatch.setattr(trainer, stage, lambda stage=stage: seen.append(stage))

    with caplog.at_level(logging.INFO, logger=PIPELINE):
        trainer.run()

    assert seen == [("seed", 11), *STAGES]
    expected = rf"Summary: ran for \d+\.\d\ds, last output saved to {re.escape(str(workspace.out))}"
    assert re.fullmatch(expected, caplog.messages[-1])


def test_without_a_seed_in_the_config_42_is_used(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    seeds: list[int] = []
    trainer = lora.ZImageLoRATrainer(workspace.lora_config(seed=None), dry_run=True, dry_run_step_time=0.0)
    monkeypatch.setattr(pipeline, "set_seed", seeds.append)

    trainer.run()

    assert seeds == [42]


def test_gradient_accumulation_steps_once_per_group_of_batches_and_the_last_group_may_be_short(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(3)
    trainer = lora.ZImageLoRATrainer(workspace.lora_config(gradient_accumulation_steps=2))
    losses = watch_losses(trainer, monkeypatch)

    trainer.run()

    assert len(losses) == 3
    assert _log_lines(workspace) == ["Step: 1/2 [50%]", "Step: 2/2 [100%]"]


def test_an_error_is_logged_with_its_traceback_and_raised_again(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)
    config = workspace.lora_config(model=None)

    with pytest.raises(KeyError, match="name_or_path"), caplog.at_level(logging.ERROR, logger=PIPELINE):
        run_lora(config)

    assert "An error occurred: 'name_or_path'" in caplog.text
    assert caplog.records[-1].exc_info is not None


def test_ctrl_c_ends_the_run_quietly_without_a_summary_or_a_checkpoint(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def interrupt() -> None:
        raise KeyboardInterrupt

    trainer = lora.ZImageLoRATrainer(workspace.lora_config())
    monkeypatch.setattr(trainer, "load_model", interrupt)

    with caplog.at_level(logging.INFO, logger=PIPELINE):
        trainer.run()

    assert "Training interrupted by user. Exiting..." in caplog.text and "Summary" not in caplog.text
    assert list(workspace.out.glob("*.safetensors")) == []


def test_a_process_that_is_not_the_main_one_saves_nothing(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = lora.ZImageLoRATrainer(workspace.lora_config())
    trainer.is_main_process = False

    trainer.save_checkpoint("final")

    assert list(workspace.out.glob("*.safetensors")) == []
    assert trainer.last_saved_path is None


def test_the_summary_names_the_output_folder_when_nothing_was_saved(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = lora.ZImageLoRATrainer(workspace.lora_config())

    with caplog.at_level(logging.INFO, logger=PIPELINE):
        trainer._log_summary(0.0)

    assert caplog.messages[-1].endswith(f"last output saved to {workspace.out}")
