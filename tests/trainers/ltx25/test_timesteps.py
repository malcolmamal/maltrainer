"""The training sigma: the official LTX shifted logit-normal (stretched in 2.3 and 2.5), or uniform."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from maltrainer.trainers.ltx25.timesteps import DISTRIBUTIONS, MODES, sample_sigma, token_shift

CPU = torch.device("cpu")


def _sigma(tokens: int = 2048, **train: Any) -> float:
    return float(sample_sigma(train, tokens, CPU))


@pytest.mark.parametrize(("tokens", "shift"), [(1024, 0.95), (4096, 2.05), (2560, 1.5), (576, 0.7896)])
def test_the_shift_is_linear_in_the_token_count_from_the_anchors_of_the_official_trainer(
    tokens: int, shift: float
) -> None:
    assert token_shift(tokens) == pytest.approx(shift, abs=1e-4)


def test_the_legacy_mode_is_sigmoid_of_a_normal_around_the_shift() -> None:
    torch.manual_seed(5)
    expected = float(torch.sigmoid(torch.randn(1) + token_shift(2048)))
    torch.manual_seed(5)

    assert _sigma(shifted_logit_mode="legacy") == pytest.approx(expected)


def test_the_stretched_mode_maps_the_percentile_range_onto_zero_to_one() -> None:
    torch.manual_seed(5)
    shift = token_shift(2048)
    base = float(torch.sigmoid(torch.randn(1) + shift))
    low = float(torch.sigmoid(torch.tensor(shift - 2.5758)))
    high = float(torch.sigmoid(torch.tensor(shift + 3.0902)))
    expected = (base - low) / (high - low)
    torch.manual_seed(5)

    assert _sigma(shifted_logit_uniform_prob=0.0) == pytest.approx(min(max(expected, 1e-3), 1.0), abs=1e-5)


def test_values_below_eps_are_reflected_back_above_it() -> None:
    torch.manual_seed(0)
    draws = [
        _sigma(shifted_logit_uniform_prob=0.0, shifted_logit_shift=-6.0, shifted_logit_eps=0.2) for _ in range(300)
    ]

    assert min(draws) >= 0.001 and sum(d < 0.2 for d in draws) < 0.1 * len(draws)


def test_the_uniform_share_replaces_that_fraction_of_the_draws_with_uniform_ones() -> None:
    torch.manual_seed(1)
    always = [_sigma(shifted_logit_uniform_prob=1.0, shifted_logit_shift=-9.0) for _ in range(200)]

    assert sum(always) / len(always) == pytest.approx(0.5, abs=0.08)  # the stretched draws would all sit near zero


def test_the_legacy_shift_can_be_overridden_and_the_spread_set() -> None:
    kwargs: dict[str, Any] = {"shifted_logit_mode": "legacy", "logit_std": 0.1}
    torch.manual_seed(3)
    high = [_sigma(shifted_logit_shift=6.0, **kwargs) for _ in range(100)]
    torch.manual_seed(3)
    low = [_sigma(shifted_logit_shift=-6.0, **kwargs) for _ in range(100)]

    assert sum(high) / 100 > 0.95 and sum(low) / 100 < 0.05


def test_uniform_is_a_plain_uniform_draw() -> None:
    torch.manual_seed(7)
    expected = float(torch.rand(1))
    torch.manual_seed(7)

    assert _sigma(timestep_distribution="uniform") == pytest.approx(expected)


def test_sigma_is_clamped_to_the_configured_range() -> None:
    draws = [_sigma(timestep_distribution="uniform", min_sigma=0.95) for _ in range(20)]
    capped = [_sigma(timestep_distribution="uniform", max_sigma=0.2) for _ in range(20)]

    assert min(draws) >= 0.95 - 1e-6 and max(capped) <= 0.2 + 1e-6 and max(capped) > 0.15


def test_unknown_names_are_refused() -> None:
    with pytest.raises(ValueError, match=r"train\.timestep_distribution must be one of .*'uniform'.*got 'beta'"):
        _sigma(timestep_distribution="beta")
    with pytest.raises(ValueError, match=r"train\.shifted_logit_mode must be one of .*'stretched'.*got 'x'"):
        _sigma(shifted_logit_mode="x")
    assert DISTRIBUTIONS == ("shifted_logit_normal", "uniform") and MODES == ("stretched", "legacy")
