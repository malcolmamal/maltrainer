"""A run resumed from its state continues exactly where a straight run would have been."""

from __future__ import annotations

import random
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
    # both draw from Python's random: the prior's recorded input per step, the half of the folder an epoch repeats
    "prior_and_fractional_repeats": (
        4,
        {"prior": {"weight": 1.0, "class_word": "person", "samples_per_image": 3}, "dataset": {"repeats": 1.5}},
    ),
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


def test_a_save_leaves_pythons_random_where_it_was(workspace: Workspace) -> None:
    # Python's random draws the next epoch's order: a save that shuffled would hand a saving run another order than
    # the same run without saves (and than the run resumed from that save)
    workspace.add_images(3)
    trainer = run_finetune(workspace.finetune_config(train={"max_steps": 2}))
    before = random.getstate()

    assert trainer.save_checkpoint(5)

    assert random.getstate() == before
