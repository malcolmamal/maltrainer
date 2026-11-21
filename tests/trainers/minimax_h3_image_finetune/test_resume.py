"""A resumed finetune ends bit for bit where an uninterrupted one does, and a resume that cannot continue says so."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from tests.trainers.minimax_h3.world import Workspace

from .world import NAME, finetune_config, run_finetune

TRAIN = {
    "finetune": {
        "blocks_per_window": 1,
        "rotate_every": 2,
        "training_adapter": False,
        "always_on": ["final_layer.video_out"],
    }
}


def _config(ws: Workspace, folder: str, max_steps: int, **sections: Any) -> dict[str, Any]:
    return finetune_config(
        ws,
        output={"path": str(ws.root / folder), "name": NAME},
        train={"max_steps": max_steps, **sections.pop("train", {})},
        **{**TRAIN, **sections},
    )


def _state(ws: Workspace, folder: str) -> Path:
    return ws.root / folder / f"{NAME}_state"


def _same_files(left: Path, right: Path) -> bool:
    a, b = load_file(str(left)), load_file(str(right))
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("stop", [2, 3, 4, 5])
def test_a_run_stopped_and_resumed_ends_bit_identical_to_one_that_was_not(workspace: Workspace, stop: int) -> None:
    workspace.add_images(3)
    run_finetune(_config(workspace, "straight", 6))

    run_finetune(_config(workspace, "split", stop))
    resumed = run_finetune(_config(workspace, "split", 6, train={"resume_from": str(_state(workspace, "split"))}))

    assert resumed.global_step == 6
    assert _same_files(
        workspace.root / "straight" / f"{NAME}.safetensors", workspace.root / "split" / f"{NAME}.safetensors"
    )
    assert _same_files(
        _state(workspace, "straight") / "weights.safetensors", _state(workspace, "split") / "weights.safetensors"
    )


def test_the_resumed_run_logs_where_it_stood(workspace: Workspace, caplog: pytest.LogCaptureFixture) -> None:
    workspace.add_images(3)
    run_finetune(_config(workspace, "split", 4))

    with caplog.at_level(logging.INFO):
        run_finetune(_config(workspace, "split", 5, train={"resume_from": str(_state(workspace, "split"))}))

    assert "Resumed at step 4 (epoch 2, sample 1 of its order)" in caplog.text
    assert "Resumed 8 trained block weights and 1 always-trained ones" in caplog.text


def test_a_resume_that_is_already_past_the_budget_says_so_and_saves_nothing(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    run_finetune(_config(workspace, "split", 3))
    saved = (workspace.root / "split" / f"{NAME}.safetensors").stat().st_mtime_ns

    with caplog.at_level(logging.WARNING):
        run_finetune(_config(workspace, "split", 3, train={"resume_from": str(_state(workspace, "split"))}))

    assert "already at step 3 of 3" in caplog.text
    assert (workspace.root / "split" / f"{NAME}.safetensors").stat().st_mtime_ns == saved


def test_a_changed_dataset_resumes_at_the_step_with_a_fresh_epoch(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    run_finetune(_config(workspace, "split", 3))
    workspace.add_images(3)

    with caplog.at_level(logging.WARNING):
        resumed = run_finetune(_config(workspace, "split", 5, train={"resume_from": str(_state(workspace, "split"))}))

    assert "The dataset changed since the saved state (2 samples then, 3 now); resuming at step 3" in caplog.text
    assert resumed.global_step == 5 and resumed.epoch >= 1


def test_a_resume_state_with_a_file_missing_is_refused_by_name(workspace: Workspace) -> None:
    workspace.add_images(2)
    run_finetune(_config(workspace, "split", 2))
    (_state(workspace, "split") / "optimizer.pt").unlink()

    with pytest.raises(FileNotFoundError, match=r"train\.resume_from: .*optimizer\.pt is missing"):
        run_finetune(_config(workspace, "split", 3, train={"resume_from": str(_state(workspace, "split"))}))


def test_a_resume_with_other_windows_than_the_saved_run_is_refused(workspace: Workspace) -> None:
    workspace.add_images(2)
    run_finetune(_config(workspace, "split", 2))

    with pytest.raises(ValueError, match="rotated windows"):
        run_finetune(
            _config(
                workspace,
                "split",
                3,
                train={"resume_from": str(_state(workspace, "split"))},
                finetune={**TRAIN["finetune"], "blocks_per_window": 2},
            )
        )


@pytest.mark.parametrize("always_on", [[], ["final_layer.video_out", "condition_proj"]])
def test_a_resume_with_other_always_trained_modules_than_the_saved_run_is_refused(
    workspace: Workspace, always_on: list[str]
) -> None:
    workspace.add_images(2)
    run_finetune(_config(workspace, "split", 2))

    with pytest.raises(ValueError, match=r"trained \['final_layer\.video_out\.weight'\].*keep finetune\.always_on"):
        run_finetune(
            _config(
                workspace,
                "split",
                3,
                train={"resume_from": str(_state(workspace, "split"))},
                finetune={**TRAIN["finetune"], "always_on": always_on},
            )
        )
