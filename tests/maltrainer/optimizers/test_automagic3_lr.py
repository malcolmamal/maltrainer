from __future__ import annotations

import logging
import math

import pytest
import torch
from torch import nn

from maltrainer.optimizers import Automagic3

LR = 1e-4


def _optimizer(params: list, history: int = 2, **options: float) -> Automagic3:
    return Automagic3(params, lr=LR, beta2=0.0, clip_threshold=10.0, polarity_history=history, **options)


def _run(optimizer: Automagic3, param: nn.Parameter, grads: list[list[float]]) -> None:
    for grad in grads:
        param.grad = torch.tensor(grad)
        optimizer.step()


def _rate(optimizer: Automagic3, param: nn.Parameter) -> float:
    return optimizer.state[param]["lr"].item()


@pytest.mark.parametrize("history", [2, 3, 4])
def test_the_rate_holds_until_the_sign_window_is_full(history: int) -> None:
    p = nn.Parameter(torch.zeros(3))
    opt = _optimizer([p], history)

    _run(opt, p, [[1.0, -1.0, 2.0]] * (history - 1))

    assert _rate(opt, p) == pytest.approx(LR)


@pytest.mark.parametrize("history", [2, 3, 4])
def test_a_consistent_direction_raises_the_rate_by_e_per_step(history: int) -> None:
    p = nn.Parameter(torch.zeros(3))
    opt = _optimizer([p], history)

    _run(opt, p, [[1.0, -1.0, 2.0]] * (history + 1))

    assert _rate(opt, p) == pytest.approx(LR * math.e**2, rel=1e-5)


@pytest.mark.parametrize("history", [2, 3, 4])
def test_signs_flipping_every_step_lower_the_rate_by_e(history: int) -> None:
    p = nn.Parameter(torch.zeros(2))
    opt = _optimizer([p], history)

    _run(opt, p, [[(-1.0) ** i, -((-1.0) ** i)] for i in range(history)])

    assert _rate(opt, p) == pytest.approx(LR / math.e, rel=1e-5)


def test_a_sign_pattern_that_is_neither_steady_nor_alternating_holds_the_rate() -> None:
    p = nn.Parameter(torch.zeros(1))
    opt = _optimizer([p], history=4)

    _run(opt, p, [[1.0], [1.0], [-1.0], [1.0]])

    assert _rate(opt, p) == pytest.approx(LR)
    assert opt.pop_stats()["signal"] == {"0": 0.0}


def test_the_vote_is_weighted_by_update_size_across_elements() -> None:
    p = nn.Parameter(torch.zeros(3))
    opt = _optimizer([p])

    _run(opt, p, [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0]])

    assert _rate(opt, p) == pytest.approx(LR * math.exp(-1.0 / 3.0), rel=1e-5)


def test_all_zero_gradients_hold_the_rate_and_the_weights() -> None:
    p = nn.Parameter(torch.ones(2))
    opt = _optimizer([p])

    _run(opt, p, [[0.0, 0.0]] * 3)

    assert _rate(opt, p) == pytest.approx(LR)
    torch.testing.assert_close(p.detach(), torch.ones(2))


def test_parameters_in_one_group_share_a_pooled_vote() -> None:
    steady, flipping = nn.Parameter(torch.zeros(3)), nn.Parameter(torch.zeros(1))
    opt = _optimizer([steady, flipping])

    for sign in (1.0, -1.0):
        steady.grad = torch.ones(3)
        flipping.grad = torch.tensor([sign])
        opt.step()

    assert _rate(opt, steady) == pytest.approx(LR * math.exp(0.5), rel=1e-5)
    assert _rate(opt, flipping) == pytest.approx(LR * math.exp(0.5), rel=1e-5)


def test_each_group_adapts_its_own_rate_and_reports_its_own_signal() -> None:
    steady, flipping = nn.Parameter(torch.zeros(2)), nn.Parameter(torch.zeros(2))
    opt = _optimizer([{"params": [steady], "name": "attn"}, {"params": [flipping]}])

    for sign in (1.0, -1.0, 1.0):
        steady.grad = torch.ones(2)
        flipping.grad = torch.tensor([sign, sign])
        opt.step()

    assert _rate(opt, steady) == pytest.approx(LR * math.e**2, rel=1e-5)
    assert _rate(opt, flipping) == pytest.approx(LR / math.e**2, rel=1e-5)
    assert opt.pop_stats() == {"signal": {"attn": 1.0, "1": -1.0}, "nonfinite": 0}
    assert opt.pop_stats() == {"signal": {}, "nonfinite": 0}


def test_the_rate_is_clamped_to_max_lr_and_min_lr() -> None:
    rising, falling = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    up = _optimizer([rising], max_lr=LR * 1.5)
    down = _optimizer([falling], min_lr=LR / 1.5)

    _run(up, rising, [[1.0], [1.0]])
    _run(down, falling, [[1.0], [-1.0]])

    assert _rate(up, rising) == pytest.approx(LR * 1.5)
    assert _rate(down, falling) == pytest.approx(LR / 1.5)


def test_the_starting_rate_is_clamped_into_the_allowed_range() -> None:
    p = nn.Parameter(torch.zeros(1))
    opt = Automagic3([p], lr=1e-9, min_lr=1e-8)

    _run(opt, p, [[1.0]])

    assert _rate(opt, p) == pytest.approx(1e-8)


def test_learning_rates_report_the_group_rate_before_and_the_state_mean_after() -> None:
    a, b = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    opt = _optimizer([{"params": [a]}, {"params": [b], "lr": 2 * LR}])
    assert opt.get_learning_rates() == [LR, 2 * LR]

    _run(opt, a, [[1.0], [1.0]])

    assert opt.get_learning_rates() == pytest.approx([LR * math.e, 2 * LR], rel=1e-5)
    assert opt.get_avg_learning_rate() == pytest.approx((LR * math.e + 2 * LR) / 2, rel=1e-5)


@pytest.mark.parametrize(("requested", "kept"), [(1, 2), (2, 2), (16, 16), (100, 64), (8.9, 8)])
def test_the_polarity_history_is_kept_between_2_and_64(requested: float, kept: int) -> None:
    opt = Automagic3([nn.Parameter(torch.zeros(1))], polarity_history=requested)

    assert opt.param_groups[0]["polarity_history"] == kept


def test_min_lr_above_max_lr_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_lr"):
        Automagic3([nn.Parameter(torch.zeros(1))], min_lr=1.0, max_lr=0.5)


def test_construction_reports_the_parameter_count_and_a_high_start_rate(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    Automagic3([nn.Parameter(torch.zeros(1000)), nn.Parameter(torch.zeros(2, 3))], lr=1e-3)
    quiet = list(caplog.messages)
    caplog.clear()
    Automagic3([nn.Parameter(torch.zeros(1))], lr=1e-2)

    assert quiet == ["Total training parameters: 1,006"]
    assert any("start lr 0.01 is high" in message for message in caplog.messages)
