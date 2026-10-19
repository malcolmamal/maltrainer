"""A group's own options survive a resume, and a group added after construction works like the others.

Automagic3 keeps the options each group was built with: a reload puts them back, so this run's
config wins over the saved one, for a group that overrides the defaults as much as for one that
does not."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.optimizers import Automagic3

SLOW = {"lr": 1e-5, "max_lr": 2e-5, "polarity_history": 2, "clip_threshold": 0.5}


def _groups(fast: nn.Parameter, slow: nn.Parameter) -> list[dict]:
    return [{"params": [fast], "name": "fast"}, {"params": [slow], "name": "slow", **SLOW}]


def _stepped(**options: object) -> tuple[Automagic3, list[nn.Parameter]]:
    params = [nn.Parameter(torch.zeros(4)), nn.Parameter(torch.zeros(4))]
    opt = Automagic3(_groups(*params), lr=1e-4, polarity_history=4, **options)
    for _ in range(3):
        for p in params:
            p.grad = torch.ones(4)
        opt.step()
    return opt, params


def _resumed(opt: Automagic3, params: list[nn.Parameter], **options: object) -> Automagic3:
    clones = [nn.Parameter(p.detach().clone()) for p in params]
    resumed = Automagic3(_groups(*clones), lr=1e-4, polarity_history=4, **options)
    resumed.load_state_dict(opt.state_dict())
    return resumed


def test_each_group_keeps_its_own_options_across_a_resume() -> None:
    opt, params = _stepped()

    resumed = _resumed(opt, params)

    fast, slow = resumed.param_groups
    assert (fast["lr"], fast["max_lr"], fast["polarity_history"], fast["clip_threshold"]) == (1e-4, 1e3, 4, 1.0)
    assert (slow["lr"], slow["max_lr"], slow["polarity_history"], slow["clip_threshold"]) == (1e-5, 2e-5, 2, 0.5)
    assert [group["name"] for group in resumed.param_groups] == ["fast", "slow"]


def test_the_new_runs_defaults_win_over_the_saved_ones_but_not_over_a_groups_override() -> None:
    opt, params = _stepped()

    resumed = _resumed(opt, params, clip_threshold=2.0, weight_decay=0.1)

    fast, slow = resumed.param_groups
    assert (fast["clip_threshold"], fast["weight_decay"]) == (2.0, 0.1)
    assert (slow["clip_threshold"], slow["weight_decay"]) == (0.5, 0.1)


def test_a_groups_own_history_length_and_rate_ceiling_apply_to_its_reloaded_state() -> None:
    opt, params = _stepped()

    resumed = _resumed(opt, params)

    fast, slow = (resumed.state[group["params"][0]] for group in resumed.param_groups)
    assert (fast["sign_history"].shape[0], slow["sign_history"].shape[0]) == (4, 2)
    assert (fast["hist_fill"], slow["hist_fill"]) == (3, 2)
    assert slow["lr"].item() <= 2e-5


def test_a_group_added_after_construction_gets_its_state_and_updates() -> None:
    first, added = nn.Parameter(torch.zeros(4)), nn.Parameter(torch.zeros(4))
    opt = Automagic3([first], lr=1e-4, polarity_history=4)

    opt.add_param_group({"params": [added], "lr": 1e-5})
    for p in (first, added):
        p.grad = torch.ones(4)
    opt.step()

    assert added.abs().sum() > 0
    assert opt.state[added]["lr"].item() == pytest.approx(1e-5, rel=1e-3)
    assert opt.get_learning_rates() == pytest.approx([1e-4, 1e-5], rel=1e-3)


def test_a_group_added_to_a_fused_optimizer_updates_during_backward() -> None:
    first, added = nn.Parameter(torch.zeros(4)), nn.Parameter(torch.zeros(4))
    opt = Automagic3([first], lr=0.5, beta2=0.0, clip_threshold=10.0, fused=True)

    opt.add_param_group({"params": [added]})
    (first.sum() + added.sum()).backward()

    assert first.abs().sum() > 0
    assert added.abs().sum() > 0
    assert len(opt._hook_handles) == 2


def test_a_group_added_after_construction_is_restored_with_its_own_options() -> None:
    first, added = nn.Parameter(torch.zeros(4)), nn.Parameter(torch.zeros(4))
    opt = Automagic3([first], lr=1e-4, polarity_history=4)
    opt.add_param_group({"params": [added], "polarity_history": 2})
    for p in (first, added):
        p.grad = torch.ones(4)
    opt.step()

    clones = [nn.Parameter(p.detach().clone()) for p in (first, added)]
    resumed = Automagic3([clones[0]], lr=1e-4, polarity_history=4)
    resumed.add_param_group({"params": [clones[1]], "polarity_history": 2})
    resumed.load_state_dict(opt.state_dict())

    assert [group["polarity_history"] for group in resumed.param_groups] == [4, 2]
    assert resumed.state[clones[1]]["sign_history"].shape[0] == 2
    assert resumed.state[clones[1]]["hist_fill"] == 1
