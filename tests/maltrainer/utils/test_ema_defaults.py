"""The defaults and the warm-up arithmetic of the EMAs, pinned (mutation testing, docs/mutation-testing.md)."""

from __future__ import annotations

import pytest
import torch

from maltrainer.utils.ema import OffloadedParameterEMA, ParameterEMA


def _param() -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.zeros(1))


def test_the_adapter_average_defaults_to_decay_098_with_warmup() -> None:
    ema = ParameterEMA([_param()])

    ema.updates = 1000  # past the ramp: (1 + n) / (10 + n) is above 0.98 from n = 441

    assert ema.decay == 0.98
    assert ema.warmup is True
    assert ema.current_decay() == 0.98


def test_the_offloaded_average_defaults_to_decay_0999_with_warmup_every_update() -> None:
    ema = OffloadedParameterEMA([("w", _param())])

    assert (ema.decay, ema.warmup, ema.every) == (0.999, True, 1)


@pytest.mark.parametrize(("updates", "expected"), [(0, 0.1), (2, 0.25), (6, 0.4375)])
def test_the_offloaded_warmup_is_one_plus_n_over_ten_plus_n(updates: int, expected: float) -> None:
    ema = OffloadedParameterEMA([("w", _param())], decay=0.5)

    ema.updates = updates  # 3 / 12 and 7 / 16 are exact, and differ from (1 - n) / (10 - n)

    assert ema.current_decay() == pytest.approx(expected)
