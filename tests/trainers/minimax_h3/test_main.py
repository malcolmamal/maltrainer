"""The H3 command line, dry run, error handling and the refusal of a batch size above one."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import pytest
import yaml

import train_minimax_h3_lora as h3

from .fakes import Fakes
from .runs import run_h3
from .world import Workspace


def _command(monkeypatch: pytest.MonkeyPatch, ws: Workspace, config: dict[str, Any], *args: str) -> None:
    path = ws.root / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_minimax_h3_lora.py", "--config", str(path), *args])


def test_main_trains_from_a_config_file_and_keeps_a_copy_of_it(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    _command(monkeypatch, workspace, workspace.config())

    h3.main()

    assert (workspace.out / "minimax_h3_lora_step_final.safetensors").exists()
    assert yaml.safe_load((workspace.out / "config.yaml").read_text(encoding="utf-8"))["lora"] == {
        "rank": 4,
        "alpha": 4,
    }


def test_a_dry_run_writes_json_checkpoints_without_loading_any_model(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    workspace.add_voice()
    config = workspace.config(dataset={"repeats": 2}, train={"num_epochs": 2, "save_every": 8})
    _command(monkeypatch, workspace, config, "--dry-run", "--dry-run-step-time", "0")

    with caplog.at_level(logging.INFO):
        h3.main()

    assert "Dry run enabled: simulating 16 steps." in caplog.text
    names = sorted(p.name for p in workspace.out.glob("*.safetensors"))
    assert names == [f"minimax_h3_lora_step_{step}.safetensors" for step in (16, 8, "final")]
    payload = json.loads((workspace.out / "minimax_h3_lora_step_8.safetensors").read_text(encoding="utf-8"))
    assert (payload["mode"], payload["kind"], payload["step"], payload["total_steps"]) == (
        "dry_run",
        "minimax_h3_lora",
        8,
        16,
    )
    assert payload["progress"]["completed_steps"] == 8 and payload["progress"]["percent"] == 50.0
    final = json.loads((workspace.out / "minimax_h3_lora_step_final.safetensors").read_text(encoding="utf-8"))
    assert final["progress"]["percent"] == 100.0 and sum(fakes.loads.values()) == 0


def test_the_dry_run_estimate_counts_files_by_kind_and_divides_by_accumulation(workspace: Workspace) -> None:
    workspace.add_images(3)
    workspace.add_voice()
    trainer = h3.MiniMaxH3LoRATrainer(
        workspace.config(dataset={"media": "image"}, train={"num_epochs": 3, "gradient_accumulation_steps": 2})
    )

    assert trainer._count_dataset_samples() == 3

    trainer.dry_run = True
    trainer.dry_run_step_time = 0.0
    trainer.run()

    assert json.loads((workspace.out / "minimax_h3_lora_step_final.safetensors").read_text())["total_steps"] == 6


def test_a_dry_run_over_an_empty_folder_still_counts_one_sample(workspace: Workspace) -> None:
    workspace.data.mkdir()

    trainer = h3.MiniMaxH3LoRATrainer(workspace.config(dataset={"repeats": 3}))

    assert trainer._count_dataset_samples() == 3


def test_a_dry_run_waits_the_step_time_between_steps(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace.add_images(2)
    waits: list[float] = []
    monkeypatch.setattr(h3.time, "sleep", waits.append)
    _command(monkeypatch, workspace, workspace.config(), "--dry-run")

    h3.main()

    assert waits == [0.025, 0.025]


def test_save_every_zero_leaves_a_dry_run_with_only_its_final_file(workspace: Workspace) -> None:
    workspace.add_images(2)
    trainer = h3.MiniMaxH3LoRATrainer(workspace.config(train={"save_every": 0}), dry_run=True, dry_run_step_time=0.0)

    trainer.run()

    assert [p.name for p in workspace.out.glob("*.safetensors")] == ["minimax_h3_lora_step_final.safetensors"]
    assert trainer.last_saved_path is not None and trainer.last_saved_path.endswith("step_final.safetensors")


@pytest.mark.parametrize("batch_size", [2, 4])
def test_a_batch_size_above_one_is_refused_in_favour_of_accumulation(workspace: Workspace, batch_size: int) -> None:
    with pytest.raises(ValueError, match=r"trains at batch_size 1; use train\.gradient_accumulation_steps"):
        h3.MiniMaxH3LoRATrainer(workspace.config(train={"batch_size": batch_size}))


@pytest.mark.parametrize("batch_size", [1, None, 0])
def test_batch_size_one_or_unset_is_accepted(workspace: Workspace, batch_size: int | None) -> None:
    assert h3.MiniMaxH3LoRATrainer(workspace.config(train={"batch_size": batch_size})).dtype is not None


def test_ctrl_c_during_setup_is_logged_and_ends_quietly(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def interrupt(_self: h3.MiniMaxH3LoRATrainer) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(h3.MiniMaxH3LoRATrainer, "precache_dataset", interrupt)

    with caplog.at_level(logging.INFO):
        run_h3(workspace.config())

    assert "Training interrupted by user. Exiting..." in caplog.text


def test_any_other_error_is_logged_with_its_traceback_and_raised_again(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(_self: h3.MiniMaxH3LoRATrainer) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(h3.MiniMaxH3LoRATrainer, "precache_dataset", fail)

    with pytest.raises(RuntimeError, match="boom"):
        run_h3(workspace.config())

    assert "An error occurred during training: boom" in caplog.text


def test_without_an_output_section_main_writes_no_config_copy_and_uses_the_default_folder(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(1)
    config = workspace.config()
    del config["output"]
    monkeypatch.chdir(workspace.root)
    _command(monkeypatch, workspace, config, "--dry-run", "--dry-run-step-time", "0")

    h3.main()

    default = workspace.root / "output"
    assert sorted(p.name for p in default.iterdir()) == ["minimax_h3_lora_step_final.safetensors", "training.log"]
