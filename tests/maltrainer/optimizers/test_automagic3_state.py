from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from maltrainer.optimizers import Automagic3


def _trained(history: int = 4, steps: int = 6, dtype: torch.dtype = torch.float32) -> tuple[nn.Parameter, Automagic3]:
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(5, dtype=dtype))
    opt = Automagic3([p, nn.Parameter(torch.zeros(2))], lr=1e-4, polarity_history=history)
    for _ in range(steps):
        p.grad = (torch.randn(5) + 0.5).to(dtype)
        opt.step()
    return p, opt


def _resumed(p: nn.Parameter, state: dict, **options: float) -> tuple[nn.Parameter, Automagic3]:
    clone = nn.Parameter(p.detach().clone())
    opt = Automagic3([clone, nn.Parameter(torch.zeros(2))], lr=1e-4, **options)
    opt.load_state_dict(state)
    return clone, opt


def test_a_saved_and_reloaded_optimizer_continues_where_it_left_off(tmp_path: Path) -> None:
    p, opt = _trained()
    torch.save(opt.state_dict(), tmp_path / "optimizer.pt")

    clone, resumed = _resumed(p, torch.load(tmp_path / "optimizer.pt"), polarity_history=4)
    grad = torch.tensor([1.0, -1.0, 0.5, 0.25, -2.0])
    for param, optimizer in ((p, opt), (clone, resumed)):
        param.grad = grad.clone()
        optimizer.step()

    torch.testing.assert_close(clone.detach(), p.detach())
    torch.testing.assert_close(resumed.state[clone]["lr"], opt.state[p]["lr"], rtol=1e-6, atol=0)
    for key in ("sign_history", "exp_avg_sq"):
        torch.testing.assert_close(resumed.state[clone][key], opt.state[p][key], rtol=0, atol=0)
    assert resumed.state[clone]["hist_idx"] == opt.state[p]["hist_idx"]


def test_the_fp32_rate_of_a_bf16_parameter_survives_a_reload() -> None:
    p, opt = _trained(dtype=torch.bfloat16)
    saved = opt.state[p]["lr"].clone()
    rounded = saved.to(torch.bfloat16).float()

    clone, resumed = _resumed(p, opt.state_dict(), polarity_history=4)

    reloaded = resumed.state[clone]["lr"]
    assert reloaded.dtype == torch.float32
    torch.testing.assert_close(reloaded, saved, rtol=1e-6, atol=0)
    assert abs(reloaded - saved) < abs(rounded - saved) / 100


def test_the_new_optimizers_options_replace_the_saved_ones() -> None:
    p, opt = _trained()

    _, resumed = _resumed(p, opt.state_dict(), polarity_history=4, clip_threshold=2.0, max_lr=0.5)

    group = resumed.param_groups[0]
    assert (group["clip_threshold"], group["max_lr"], group["min_lr"]) == (2.0, 0.5, 1e-8)


def test_reloaded_rates_in_a_group_are_set_to_their_geometric_median() -> None:
    params = [nn.Parameter(torch.zeros(1)) for _ in range(3)]
    opt = Automagic3(params, lr=1e-4)
    for p in params:
        p.grad = torch.ones(1)
    opt.step()
    state = opt.state_dict()
    for index, rate in enumerate((1e-2, 1e-6, 1e-3)):
        state["state"][index]["lr"] = torch.tensor(rate)

    resumed = Automagic3([nn.Parameter(torch.zeros(1)) for _ in range(3)], lr=1e-4)
    resumed.load_state_dict(state)

    assert resumed.get_learning_rates() == pytest.approx([1e-3], rel=1e-5)


def test_a_reloaded_pinned_rate_stays_exactly_on_the_pin() -> None:
    p, opt = _trained()
    pin = torch.tensor(3e-5)
    state = opt.state_dict()
    state["state"][0]["lr"] = pin.clone()

    clone, resumed = _resumed(p, state, polarity_history=4, min_lr=pin.item(), max_lr=pin.item())

    assert resumed.state[clone]["lr"].item() == pin.item()


@pytest.mark.parametrize(
    ("change", "options"),
    [
        ({}, {"polarity_history": 8}),
        ({"hist_idx": 4}, {"polarity_history": 4}),
        ({"hist_idx": torch.tensor(1)}, {"polarity_history": 4}),
        ({"hist_fill": 5}, {"polarity_history": 4}),
        ({"sign_history": [0, 1]}, {"polarity_history": 4}),
    ],
)
def test_an_unusable_sign_history_is_restarted_empty(change: dict, options: dict) -> None:
    p, opt = _trained()
    state = opt.state_dict()
    state["state"][0].update(change)

    clone, resumed = _resumed(p, state, **options)

    st = resumed.state[clone]
    history = options["polarity_history"]
    torch.testing.assert_close(st["sign_history"], torch.zeros((history, 1), dtype=torch.uint8))
    assert (st["hist_idx"], st["hist_fill"]) == (0, 0)


def test_a_restarted_history_waits_a_full_window_before_voting_again() -> None:
    p, opt = _trained()
    clone, resumed = _resumed(p, opt.state_dict(), polarity_history=2)
    rate = resumed.state[clone]["lr"].item()

    clone.grad = torch.ones(5)
    resumed.step()

    assert resumed.state[clone]["lr"].item() == rate
    assert resumed.state[clone]["hist_fill"] == 1


def test_a_state_entry_without_a_rate_gets_an_empty_history_and_reports_the_group_rate() -> None:
    p = nn.Parameter(torch.zeros(3))
    opt = Automagic3([p])
    state = {"state": {0: {"step": 3}}, "param_groups": opt.state_dict()["param_groups"]}

    opt.load_state_dict(state)

    assert "lr" not in opt.state[p]
    torch.testing.assert_close(opt.state[p]["sign_history"], torch.zeros((8, 1), dtype=torch.uint8))
    assert opt.get_learning_rates() == [1e-6]
