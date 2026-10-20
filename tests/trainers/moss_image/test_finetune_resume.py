"""A run resumed from its state continues exactly where a straight run would be (docs/moss-image-finetuning.md)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from .runs import optimizer_state, run_finetune, same_values
from .world import Workspace, named_weights

STEPS, SPLIT = 7, 4
Scenario = Callable[[Workspace], dict[str, Any]]
ADAPTIVE = {"automagic_min_lr": 1e-8, "automagic_max_lr": 1e3, "automagic_polarity_history": 2}
SCENARIOS: dict[str, Scenario] = {
    "plain": lambda _ws: {},
    "ema_with_adaptive_rates": lambda _ws: {"train": {**ADAPTIVE, "ema_decay": 0.9, "ema_every": 2}},
    "unfused_with_dropout": lambda _ws: {
        "train": {"automagic_fused": False, "caption_dropout": 0.5, "timestep_uniform_mix": 0.5}
    },
    "prior_and_regularisation": lambda ws: {
        "dataset": {"reg_path": str(ws.root / "reg")},
        "prior": {"weight": 1.0, "samples_per_image": 2},
    },
}


def _runs(ws: Workspace, scenario: dict[str, Any]) -> tuple[Any, Any, Any]:
    ws.add_images("p", 3)
    ws.add_images(ws.root / "reg", 2, caption="a photo of a person")

    def at(folder: str, **train: Any) -> dict[str, Any]:
        return ws.config(scenario, output={"path": str(ws.root / folder)}, train=train)

    straight = run_finetune(at("straight", max_steps=STEPS))
    split = run_finetune(at("split", max_steps=SPLIT))
    resumed = run_finetune(at("split", max_steps=STEPS, resume_from=str(ws.root / "split" / "ft_state")))
    return straight, split, resumed


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_a_resumed_run_ends_bit_for_bit_where_the_straight_run_ends(workspace: Workspace, name: str) -> None:
    straight, split, resumed = _runs(workspace, SCENARIOS[name](workspace))

    assert split.global_step == SPLIT
    assert (resumed.global_step, resumed.epoch, resumed.position) == (STEPS, straight.epoch, straight.position)
    assert same_values(named_weights(resumed.transformer), named_weights(straight.transformer))
    assert same_values(optimizer_state(resumed.optimizer), optimizer_state(straight.optimizer))
    assert (resumed.ema is None) == (straight.ema is None)
    if straight.ema is not None:
        assert (resumed.ema.updates, resumed.ema.calls) == (straight.ema.updates, straight.ema.calls)
        assert same_values(resumed.ema.averaged(), straight.ema.averaged())
