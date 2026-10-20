"""Resuming a finetune from its state: what is refused, what is logged and what happens to a changed dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from train_krea2_finetune import STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER

from .runs import run_finetune
from .world import Workspace, write_checkpoint, write_image

STATE_FILES = {STATE_MODEL, STATE_OPTIMIZER, STATE_TRAINER}


def _state(ws: Workspace) -> Path:
    return ws.out / "ft_state"


def _counters(ws: Workspace) -> dict[str, Any]:
    return torch.load(_state(ws) / STATE_TRAINER, map_location="cpu", weights_only=False)


def _resume(ws: Workspace, **sections: Any) -> Any:
    return run_finetune(
        ws.finetune_config(train={"resume_from": str(_state(ws)), **sections.pop("train", {})}, **sections)
    )


@pytest.mark.parametrize("missing", sorted(STATE_FILES))
def test_resuming_from_an_incomplete_state_names_the_missing_file(workspace: Workspace, missing: str) -> None:
    workspace.add_images(1)
    run_finetune(workspace.finetune_config())
    (_state(workspace) / missing).unlink()

    with pytest.raises(FileNotFoundError, match=rf"train\.resume_from: .*{missing} is missing"):
        _resume(workspace)


def test_resuming_with_other_windows_is_refused(workspace: Workspace) -> None:
    workspace.add_images(1)
    run_finetune(workspace.finetune_config())

    with pytest.raises(
        ValueError, match=r"The resumed run rotated windows \['attention', 'mlp.gate'.*keep finetune.windows"
    ):
        _resume(workspace, finetune={"windows": [{"name": "everything", "prefixes": ["attn.", "mlp."]}]})


def test_a_resume_logs_where_it_continues_and_loads_the_saved_weights_as_the_base(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    run_finetune(workspace.finetune_config(train={"max_steps": 2}))

    with caplog.at_level(logging.INFO):
        trainer = _resume(workspace, train={"max_steps": 4})

    assert f"Resuming from {_state(workspace)}" in caplog.text
    assert "Resumed at step 2 (epoch 1, sample 2 of its order)" in caplog.text
    assert trainer.run_metadata["base_load"] == {"source_fp8_layers": 0, "quantized_on_load": 16, "fp8_linears": 16}


def test_the_fp8_start_warning_is_not_repeated_on_a_resume(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    write_checkpoint(workspace.checkpoint, fp8=True)
    workspace.add_images(3)
    run_finetune(workspace.finetune_config(train={"max_steps": 2}))
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        _resume(workspace, train={"max_steps": 4})

    assert "stores its blocks in fp8" not in caplog.text


def test_a_changed_dataset_resumes_at_the_step_with_a_fresh_epoch(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    run_finetune(workspace.finetune_config(train={"max_steps": 2}))
    write_image(workspace.data / "extra.png", "a photo of zed")

    with caplog.at_level(logging.WARNING):
        trainer = _resume(workspace, train={"max_steps": 4})

    assert (
        "The dataset changed since the saved state (3 samples then, 4 now); resuming at step 2 with a fresh epoch."
        in caplog.text
    )
    assert (trainer.global_step, trainer.epoch) == (4, 1)


def test_a_run_that_is_already_at_its_last_step_says_so_and_saves_nothing_new(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    run_finetune(workspace.finetune_config(train={"max_steps": 2}))
    before = (workspace.out / "ft.safetensors").read_bytes()

    with caplog.at_level(logging.WARNING):
        trainer = _resume(workspace, train={"max_steps": 2})

    assert "The resumed run is already at step 2 of 2; raise train.max_steps to continue." in caplog.text
    assert trainer.global_step == 2 and (workspace.out / "ft.safetensors").read_bytes() == before


def test_a_state_saved_at_the_start_of_an_epoch_keeps_its_epoch_when_the_dataset_changed(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    run_finetune(workspace.finetune_config(train={"max_steps": 2}))
    counters = _counters(workspace)
    torch.save({**counters, "position": 0}, _state(workspace) / STATE_TRAINER)
    write_image(workspace.data / "extra.png", "a photo of zed")

    with caplog.at_level(logging.WARNING):
        trainer = _resume(workspace, train={"max_steps": 3})

    assert "The dataset changed since the saved state" in caplog.text
    assert (trainer.global_step, trainer.epoch, trainer.position) == (3, 0, 1)
