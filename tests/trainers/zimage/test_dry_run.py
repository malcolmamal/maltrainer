"""``--dry-run``: both trainers write progress files for the steps a real run would take, without models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

import train_zimage_embedding as embedding
import train_zimage_lora as lora

from .world import Workspace

KINDS = pytest.mark.parametrize("kind", ["lora", "embedding"])


def _trainer(ws: Workspace, kind: str, sections: dict[str, Any], step_time: float = 0.0) -> Any:
    if kind == "lora":
        return lora.ZImageLoRATrainer(ws.lora_config(**sections), dry_run=True, dry_run_step_time=step_time)
    return embedding.ZImageEmbeddingTrainer(ws.embedding_config(**sections), dry_run=True, dry_run_step_time=step_time)


def _written(ws: Workspace, kind: str) -> dict[str, dict[str, Any]]:
    files = sorted(ws.out.glob(f"zimage_{kind}_step_*.safetensors"))
    return {
        p.name.removeprefix(f"zimage_{kind}_step_").removesuffix(".safetensors"): json.loads(p.read_text())
        for p in files
    }


@KINDS
def test_a_dry_run_writes_a_progress_file_every_save_every_steps_and_a_final_one(
    workspace: Workspace, kind: str
) -> None:
    workspace.add_images(3)
    trainer = _trainer(workspace, kind, {"train": {"num_epochs": 2, "save_every": 4}})

    trainer.run()

    written = _written(workspace, kind)
    assert list(written) == ["4", "final"]
    assert trainer.last_saved_path == str(workspace.out / f"zimage_{kind}_step_final.safetensors")


@KINDS
def test_the_step_count_follows_images_repeats_batch_size_and_accumulation(workspace: Workspace, kind: str) -> None:
    workspace.add_images(5)
    sections = {
        "dataset": {"repeats": 2},
        "train": {"num_epochs": 3, "batch_size": 2, "gradient_accumulation_steps": 2, "save_every": 0},
    }

    _trainer(workspace, kind, sections).run()

    final = _written(workspace, kind)["final"]
    assert (final["steps_per_epoch"], final["num_epochs"], final["total_steps"]) == (3, 3, 9)


@KINDS
def test_the_payload_records_the_config_the_progress_and_the_derived_settings(workspace: Workspace, kind: str) -> None:
    workspace.add_images(2)
    trainer = _trainer(workspace, kind, {"train": {"num_epochs": 2, "save_every": 2}})

    trainer.run()

    mid = _written(workspace, kind)["2"]
    assert (mid["mode"], mid["kind"], mid["step"], mid["config"]) == ("dry_run", kind, 2, trainer.config)
    assert (mid["progress"]["completed_steps"], mid["progress"]["percent"]) == (2, 50.0)
    assert mid["derived"] == {
        "dataset_path": str(workspace.data),
        "dataset_resolution": 16,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_every": 2,
        "dtype": "float32",
        "mixed_precision": "no",
        "output_path": str(workspace.out),
        **({"trigger": "sks", "num_vectors": 1} if kind == "embedding" else {}),
    }
    final = _written(workspace, kind)["final"]
    assert (final["step"], final["progress"]["percent"], final["progress"]["eta_seconds"]) == ("final", 100.0, 0.0)


@KINDS
def test_each_simulated_step_sleeps_for_the_step_time_and_the_eta_counts_the_steps_left(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    workspace.add_images(4)
    trainer = _trainer(workspace, kind, {"train": {"save_every": 1}}, step_time=0.5)

    trainer.run()

    assert sleeps == [0.5] * 4
    assert [_written(workspace, kind)[s]["progress"]["eta_seconds"] for s in ("1", "3")] == [1.5, 0.5]
    assert _written(workspace, kind)["1"]["progress"]["step_time_seconds"] == 0.5


@KINDS
def test_zero_epochs_or_batch_size_count_as_one_and_a_missing_dataset_as_one_image(
    workspace: Workspace, kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = _trainer(workspace, kind, {"train": {"num_epochs": 0, "batch_size": 0, "save_every": 0}})

    with caplog.at_level(logging.WARNING):
        trainer.run()

    final = _written(workspace, kind)["final"]
    assert (final["num_epochs"], final["steps_per_epoch"], final["total_steps"]) == (1, 1, 1)
    assert f"Dataset path does not exist: {workspace.data}" in caplog.text


@KINDS
def test_a_dry_run_logs_its_plan_and_a_summary_naming_the_last_file(
    workspace: Workspace, caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    workspace.add_images(2)
    trainer = _trainer(workspace, kind, {"train": {"num_epochs": 3}})

    with caplog.at_level(logging.INFO):
        trainer.run()

    assert "Simulating 6 steps (epochs=3, steps/epoch=2, batch_size=1, grad_accum=1, step_time=0.00s)." in caplog.text
    assert "Summary: ran for" in caplog.text
    assert f"last output saved to {Path(trainer.last_saved_path)}" in caplog.text
    assert not trainer.__dict__.keys() & {"transformer", "vae", "text_encoder"}
