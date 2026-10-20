"""The learning-rate schedule and the progress line: warmup, and what each optimizer reports as its rate."""

from __future__ import annotations

import re
from typing import Any

import pytest

from .runs import prepared_lora, progress_lines, run_lora
from .world import Workspace


def test_warmup_scales_the_rate_up_over_the_first_steps_and_without_it_there_is_no_scheduler(
    workspace: Workspace,
) -> None:
    workspace.add_images(2)

    plain = prepared_lora(workspace.lora_config(train={"optimizer": "adamw"}))
    warmed = run_lora(workspace.lora_config(train={"optimizer": "adamw", "learning_rate_warmup_steps": 10}))

    assert plain.lr_scheduler is None
    assert warmed.lr_scheduler is not None
    assert warmed.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 * 4 / 10)


@pytest.mark.parametrize(
    ("train", "expected"),
    [
        ({}, r"LR: 1\.00e-03 \(automagic3: attn 1\.00e-03, mlp 1\.00e-03\)$"),
        ({"optimizer": "prodigy"}, r"LR: 1\.00e-06 \(mult 1\.000 x d 1\.00e-06\)$"),
        ({"optimizer": "adamw"}, r"LR: 1\.00e-03$"),
    ],
)
def test_the_progress_line_shows_the_rate_the_optimizer_actually_uses(
    workspace: Workspace, train: dict[str, Any], expected: str
) -> None:
    workspace.add_images(1)

    run_lora(workspace.lora_config(train={"max_steps": 1, **train}))

    assert re.search(r"^.*Step: 1/1 \[100%\], Loss: \d\.\d{6}, Epoch: 1/1, " + expected, progress_lines(workspace)[0])
