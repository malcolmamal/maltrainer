"""The embedding trainer's run(): its stages, gradient accumulation, interruption and failure handling."""

from __future__ import annotations

import logging
import re
from typing import Any

import pytest

import train_zimage_embedding as embedding

from .runs import run_embedding, watch_losses
from .world import Workspace

STAGES = ["load_model", "create_embedding", "create_dataset", "setup_optimizer", "train"]


def test_the_stages_run_in_order_between_seeding_and_the_summary(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[Any] = []
    trainer = embedding.ZImageEmbeddingTrainer(workspace.embedding_config(seed=11))
    monkeypatch.setattr(embedding, "set_seed", lambda seed: seen.append(("seed", seed)))
    for stage in STAGES:
        monkeypatch.setattr(trainer, stage, lambda stage=stage: seen.append(stage))

    with caplog.at_level(logging.INFO, logger="train_zimage_embedding"):
        trainer.run()

    assert seen == [("seed", 11), *STAGES]
    expected = rf"Summary: ran for \d+\.\d\ds, last output saved to {re.escape(str(workspace.out))}"
    assert re.fullmatch(expected, caplog.messages[-1])


def test_without_a_seed_in_the_config_42_is_used(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    seeds: list[int] = []
    config = workspace.embedding_config(seed=None)
    trainer = embedding.ZImageEmbeddingTrainer(config, dry_run=True, dry_run_step_time=0.0)
    monkeypatch.setattr(embedding, "set_seed", seeds.append)

    trainer.run()

    assert seeds == [42]


def test_gradient_accumulation_runs_every_batch_but_counts_a_step_only_when_the_group_is_done(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(3)
    config = workspace.embedding_config(gradient_accumulation_steps=2, train={"save_every": 2})
    trainer = embedding.ZImageEmbeddingTrainer(config)
    losses = watch_losses(trainer, monkeypatch)

    trainer.run()

    assert len(losses) == 3
    assert sorted(p.name for p in workspace.out.glob("*.safetensors")) == [
        "zimage_embedding_step_2.safetensors",
        "zimage_embedding_step_final.safetensors",
    ]


def test_an_error_is_logged_with_its_traceback_and_raised_again(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)
    config = workspace.embedding_config()
    del config["model"]["name_or_path"]

    with pytest.raises(KeyError, match="name_or_path"), caplog.at_level(logging.ERROR, logger="train_zimage_embedding"):
        run_embedding(config)

    assert "An error occurred: 'name_or_path'" in caplog.text
    assert caplog.records[-1].exc_info is not None


def test_ctrl_c_ends_the_run_quietly_without_a_summary_or_a_checkpoint(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def interrupt() -> None:
        raise KeyboardInterrupt

    trainer = embedding.ZImageEmbeddingTrainer(workspace.embedding_config())
    monkeypatch.setattr(trainer, "load_model", interrupt)

    with caplog.at_level(logging.INFO, logger="train_zimage_embedding"):
        trainer.run()

    assert "Training interrupted by user. Exiting..." in caplog.text and "Summary" not in caplog.text
    assert not workspace.out.exists()


def test_a_process_that_is_not_the_main_one_saves_nothing(workspace: Workspace) -> None:
    trainer = embedding.ZImageEmbeddingTrainer(workspace.embedding_config())
    trainer.is_main_process = False

    trainer.save_checkpoint("final")

    assert not workspace.out.exists() and trainer.last_saved_path is None


def test_the_summary_names_the_output_folder_when_nothing_was_saved(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    trainer = embedding.ZImageEmbeddingTrainer(workspace.embedding_config())

    with caplog.at_level(logging.INFO, logger="train_zimage_embedding"):
        trainer._log_summary(0.0)

    assert caplog.messages[-1].endswith(f"last output saved to {workspace.out}")
