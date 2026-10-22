"""When the real run writes checkpoints: every save_every steps, never twice for the last step, and 0 means never."""

from __future__ import annotations

import pytest

from .runs import run_h3
from .world import Workspace

PREFIX, SUFFIX = "minimax_h3_lora_step_", ".safetensors"


def _saved(ws: Workspace) -> list[str]:
    return sorted(path.name.removeprefix(PREFIX).removesuffix(SUFFIX) for path in ws.out.glob("*" + SUFFIX))


@pytest.mark.parametrize(
    ("save_every", "expected"),
    [(2, ["2", "final"]), (1, ["1", "2", "3", "final"]), (0, ["final"]), (10, ["final"])],
)
def test_a_numbered_checkpoint_is_written_every_save_every_steps_and_the_last_step_only_once(
    workspace: Workspace, save_every: int, expected: list[str]
) -> None:
    workspace.add_images(2)

    run_h3(workspace.config(train={"num_epochs": 2, "gradient_accumulation_steps": 1, "save_every": save_every}))

    # 2 images x 2 epochs = 4 steps; a numbered save at step 4 would repeat the final one
    assert _saved(workspace) == expected
