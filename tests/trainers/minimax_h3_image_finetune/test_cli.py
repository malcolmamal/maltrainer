"""The finetune's command line, its refusals up front, and a dry run that loads nothing."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from accelerate import Accelerator

import train_minimax_h3_image_finetune as finetune
from tests.trainers.minimax_h3.fakes import Fakes
from tests.trainers.minimax_h3.world import Workspace
from train_minimax_h3_image_finetune import MiniMaxH3ImageFinetuneTrainer

from .world import NAME, finetune_config


def _command(monkeypatch: pytest.MonkeyPatch, ws: Workspace, config: dict[str, Any], *args: str) -> None:
    path = ws.root / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_minimax_h3_image_finetune.py", "--config", str(path), *args])


def test_main_trains_from_a_config_file(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace.add_images(2)
    _command(monkeypatch, workspace, finetune_config(workspace, output={"name": NAME}, train={"max_steps": 2}))

    finetune.main()

    assert (workspace.out / f"{NAME}.safetensors").exists() and (workspace.out / f"{NAME}_state").is_dir()


def test_resume_from_on_the_command_line_overrides_the_config(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    _command(monkeypatch, workspace, finetune_config(workspace, output={"name": NAME}, train={"max_steps": 2}))
    finetune.main()
    state = str(workspace.out / f"{NAME}_state")
    _command(
        monkeypatch,
        workspace,
        finetune_config(workspace, output={"name": NAME}, train={"max_steps": 3}),
        "--resume-from",
        state,
    )

    with caplog.at_level(logging.INFO):
        finetune.main()

    assert "Resumed at step 2 (epoch 1, sample 2 of its order)" in caplog.text


def test_a_dry_run_plans_from_the_folder_and_writes_progress_records_only(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(3)
    config = finetune_config(workspace, output={"path": str(workspace.root / "dry")}, train={"num_epochs": 2})
    _command(monkeypatch, workspace, config, "--dry-run", "--dry-run-step-time", "0")

    finetune.main()

    record = json.loads((workspace.root / "dry" / "minimax_h3_image_finetune_step_final.safetensors").read_text())
    assert record["mode"] == "dry_run" and record["total_steps"] == 6
    assert fakes.loads == {"video": 0, "audio": 0, "text": 0, "transformer": 0}


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ({"train": {"gradient_accumulation_steps": 2}}, "gradient_accumulation_steps must be 1"),
        ({"finetune": {"rotate_every": 0}}, "rotate_every must be at least 1"),
    ],
)
def test_setups_the_finetune_cannot_run_are_refused_before_anything_loads(
    workspace: Workspace, sections: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MiniMaxH3ImageFinetuneTrainer(finetune_config(workspace, **sections))


def test_only_automagic3_runs_a_finetune(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = MiniMaxH3ImageFinetuneTrainer(finetune_config(workspace, train={"optimizer": "adamw"}))

    with pytest.raises(ValueError, match="supports automagic3 only"):
        trainer.run()


def test_a_finetune_runs_on_one_process_only(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Accelerator, "num_processes", property(lambda _self: 2))

    with pytest.raises(ValueError, match="runs on one GPU"):
        MiniMaxH3ImageFinetuneTrainer(finetune_config(workspace))


def test_a_run_without_images_says_so(workspace: Workspace) -> None:
    Path(workspace.data).mkdir(parents=True, exist_ok=True)
    trainer = MiniMaxH3ImageFinetuneTrainer(finetune_config(workspace))

    with pytest.raises(ValueError, match="No training media"):
        trainer.run()


def test_a_dataset_whose_files_all_failed_to_load_stops_the_loop(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MiniMaxH3ImageFinetuneTrainer, "precache_dataset", lambda _self: None)

    with pytest.raises(ValueError, match="No usable images"):
        MiniMaxH3ImageFinetuneTrainer(finetune_config(workspace)).run()
