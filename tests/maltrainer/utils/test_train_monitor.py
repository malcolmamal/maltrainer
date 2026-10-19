from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Dict

import pytest
import torch

from maltrainer.utils import train_monitor
from maltrainer.utils.train_monitor import NonFiniteLoss, TrainMonitor, WeightDrift


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now


class _Optimizer:
    def __init__(self, signal: Dict[str, float], nonfinite: int) -> None:
        self.stats = {"signal": signal, "nonfinite": nonfinite}

    def pop_stats(self) -> dict:
        return self.stats


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(train_monitor, "time", SimpleNamespace(time=fake.time))
    return fake


def _set(param: torch.nn.Parameter, *values: float) -> None:
    with torch.no_grad():
        param.copy_(torch.tensor(values))


def test_drift_is_the_relative_change_grouped_by_label() -> None:
    attn, mlp_a, mlp_b = (torch.nn.Parameter(torch.tensor(v)) for v in ([3.0, 4.0], [3.0, 0.0], [0.0, 4.0]))
    drift = WeightDrift([("attn", attn), ("mlp", mlp_a), ("mlp", mlp_b)])
    _set(attn, 6.0, 8.0)
    _set(mlp_a, 0.0, 0.0)

    measured = drift.measure()

    assert measured == pytest.approx({"attn": 1.0, "mlp": 0.6})
    assert drift.size_mb == 24 / 1024**2


def test_the_drift_summary_reports_its_growth_exponent_from_the_second_call() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0, 0.0]))
    drift = WeightDrift([("attn", param)])
    _set(param, 1.0, 0.1)
    first = drift.summary(10)
    _set(param, 1.0, 0.2)

    second = drift.summary(20)

    assert first == "attn 1.0e-01"
    assert second == "attn 2.0e-01 (t^1.00)"


def test_no_exponent_is_shown_while_nothing_has_drifted() -> None:
    drift = WeightDrift([("attn", torch.nn.Parameter(torch.tensor([1.0])))])

    assert [drift.summary(10), drift.summary(20)] == ["attn 0.0e+00", "attn 0.0e+00"]


def test_the_summary_averages_the_window_and_splits_it_into_noise_bands(clock: _Clock) -> None:
    monitor = TrainMonitor(every=10)
    for loss, sigma in [(0.2, 0.1), (0.4, 0.3), (0.6, 0.5), (0.8, 0.9), (1.0, 1.5)]:
        monitor.record(loss, sigma, prior=0.001)
    monitor.steps = 2
    clock.now += 7.0
    monitor.exclude(3.0)

    summary = monitor.summary(10, 100)

    assert (
        summary
        == "step 10/100 | loss 0.6000 (low 0.300 n=2, mid 0.600 n=1, high 0.800 n=1) | prior 0.00100 | 2.00 s/step"
    )


def test_an_empty_window_shows_only_the_step() -> None:
    assert TrainMonitor().summary(0, 10) == "step 0/10"


def test_non_finite_priors_are_left_out_of_the_average() -> None:
    monitor = TrainMonitor()
    monitor.record(0.5, 0.6, prior=math.nan)
    monitor.record(0.5, 0.6, prior=0.004)

    assert monitor.priors == [0.004]


def test_optimizer_statistics_are_read_through_a_wrapper(caplog: pytest.LogCaptureFixture) -> None:
    wrapped = SimpleNamespace(optimizer=_Optimizer({"attn": 0.12, "mlp": -0.05}, nonfinite=3))

    with caplog.at_level(logging.WARNING, logger="maltrainer.utils.train_monitor"):
        summary = TrainMonitor().summary(5, 10, wrapped)

    assert summary == "step 5/10 | signal attn +0.12 mlp -0.05 | nonfinite 3"
    assert caplog.messages == ["[monitor] 3 NaN / inf gradient values since the last summary (zeroed)"]


def test_an_optimizer_without_signal_still_reports_its_nonfinite_count() -> None:
    assert TrainMonitor().summary(5, 10, _Optimizer({}, nonfinite=0)) == "step 5/10 | nonfinite 0"


def test_the_drift_joins_the_summary() -> None:
    drift = WeightDrift([("attn", torch.nn.Parameter(torch.tensor([1.0])))])

    assert TrainMonitor(drift=drift).summary(1, 10) == "step 1/10 | drift attn 0.0e+00"


def test_a_summary_is_logged_every_interval_and_at_the_last_step(
    clock: _Clock, caplog: pytest.LogCaptureFixture
) -> None:
    monitor = TrainMonitor(every=2)

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.train_monitor"):
        for step in range(1, 4):
            monitor.record(0.5, 0.2)
            clock.now += 1.5
            monitor.step_done(step, total=3)

    assert caplog.messages == [
        "[monitor] step 2/3 | loss 0.5000 (low 0.500 n=2) | 1.50 s/step",
        "[monitor] step 3/3 | loss 0.5000 (low 0.500 n=1) | 1.50 s/step",
    ]
    assert (monitor.steps, monitor.steps_total, monitor.losses) == (0, 3, [])


def test_a_loss_that_stays_non_finite_stops_the_run(caplog: pytest.LogCaptureFixture) -> None:
    monitor = TrainMonitor(stop_after=2)

    with caplog.at_level(logging.ERROR, logger="maltrainer.utils.train_monitor"):
        monitor.record(math.inf, 0.5)
        with pytest.raises(NonFiniteLoss, match="NaN / inf 2 steps in a row"):
            monitor.record(math.nan, 0.5)

    assert caplog.messages == ["[monitor] the loss is inf (1 in a row)", "[monitor] the loss is nan (2 in a row)"]
    assert monitor.losses == []


def test_a_finite_loss_resets_the_non_finite_count() -> None:
    monitor = TrainMonitor(stop_after=2)
    monitor.record(math.nan, 0.5)

    monitor.record(0.3, 0.5)
    monitor.record(math.nan, 0.5)

    assert monitor.bad_in_a_row == 1
    assert monitor.losses == [0.3]
