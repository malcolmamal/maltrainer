"""A run resumed from its state continues exactly where a straight run would have been."""

from __future__ import annotations

from typing import Any

import pytest

from train_krea2_finetune import Krea2FinetuneTrainer

from .runs import final_weights, leaves, run_finetune
from .world import Workspace, same_values

STEPS = 7
SCENARIOS: dict[str, tuple[int, dict[str, Any]]] = {
    "mid_epoch": (4, {}),
    "epoch_boundary": (3, {}),
    "dropout_and_uniform_timesteps": (4, {"train": {"caption_dropout": 0.5, "timestep_distribution": "uniform"}}),
    "one_step_windows_without_text_fusion": (5, {"finetune": {"rotate_every": 1, "always_on": []}}),
}


def _optimizers(trainer: Krea2FinetuneTrainer) -> dict[str, Any]:
    always = None if trainer.always_optimizer is None else trainer.always_optimizer.state_dict()
    return leaves({"windows": trainer.rotator.state_dict(), "always": always})


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_a_resumed_run_ends_bit_for_bit_where_the_straight_run_ends(workspace: Workspace, name: str) -> None:
    split_at, scenario = SCENARIOS[name]
    workspace.add_images(3)

    def at(folder: str, **train: Any) -> dict[str, Any]:
        sections = {**scenario, "train": {**scenario.get("train", {}), **train}}
        return workspace.finetune_config(**sections, output={"path": str(workspace.root / folder)})

    straight = run_finetune(at("straight", max_steps=STEPS))
    split = run_finetune(at("split", max_steps=split_at))
    state = str(workspace.root / "split" / "ft_state")
    resumed = run_finetune(at("split", max_steps=STEPS, resume_from=state))

    assert split.global_step == split_at
    assert (resumed.global_step, resumed.epoch, resumed.position) == (STEPS, straight.epoch, straight.position)
    assert same_values(final_weights(workspace.root / "split"), final_weights(workspace.root / "straight"))
    assert same_values(_optimizers(resumed), _optimizers(straight))
