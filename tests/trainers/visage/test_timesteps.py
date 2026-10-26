"""The training sigma: the release recipe's token-dependent logit-normal, uniform or logit-normal on request."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from maltrainer.trainers.visage.timesteps import DISTRIBUTIONS, logit_mean, sample_sigma

CPU = torch.device("cpu")


def _sigma(tokens: int = 256, **train: Any) -> float:
    return float(sample_sigma(train, tokens, CPU))


@pytest.mark.parametrize(("tokens", "mean"), [(256, math.log(2)), (64, 0.0), (1024, math.log(4)), (16, -math.log(2))])
def test_the_logit_mean_is_the_log_of_the_square_root_of_the_token_ratio(tokens: int, mean: float) -> None:
    assert logit_mean(tokens) == pytest.approx(mean)


def test_the_reference_token_count_can_be_moved() -> None:
    assert logit_mean(256, 256) == 0.0


def test_the_default_is_sigmoid_of_a_normal_around_the_release_shift() -> None:
    torch.manual_seed(5)
    expected = float(torch.sigmoid(torch.randn(1) + math.log(2)))
    torch.manual_seed(5)

    assert _sigma() == pytest.approx(expected)


def test_the_draw_moves_toward_noise_as_the_image_gets_bigger() -> None:
    torch.manual_seed(5)
    small = _sigma(tokens=64)
    torch.manual_seed(5)
    large = _sigma(tokens=1024)

    assert large > small


def test_the_timestep_ref_tokens_setting_moves_the_shift() -> None:
    torch.manual_seed(5)
    expected = float(torch.sigmoid(torch.randn(1)))
    torch.manual_seed(5)

    assert _sigma(timestep_ref_tokens=256) == pytest.approx(expected)


def test_the_logit_std_applies_to_the_release_recipe_too() -> None:
    assert _sigma(timestep_logit_std=0.0) == pytest.approx(2 / 3)  # sigmoid(ln 2)


def test_uniform_and_logit_normal_are_available_with_the_shift_applied() -> None:
    torch.manual_seed(7)
    base = float(torch.rand(1))
    torch.manual_seed(7)

    assert _sigma(timestep_distribution="uniform", timestep_shift=3.0) == pytest.approx(3 * base / (1 + 2 * base))
    assert _sigma(timestep_distribution="logit_normal", timestep_logit_std=0.0) == 0.5


@pytest.mark.parametrize(("limits", "expected"), [({"min_sigma": 0.95}, 0.95), ({"max_sigma": 0.2}, 0.2)])
def test_sigma_is_clamped_to_the_configured_range(limits: dict[str, float], expected: float) -> None:
    sigma = _sigma(timestep_distribution="logit_normal", timestep_logit_std=0.0, timestep_logit_mean=0.0, **limits)

    assert sigma == pytest.approx(expected)


def test_an_unknown_distribution_is_refused() -> None:
    with pytest.raises(ValueError, match=r"train\.timestep_distribution must be one of .*'visage'.*got 'beta'"):
        _sigma(timestep_distribution="beta")
    assert DISTRIBUTIONS == ("visage", "uniform", "logit_normal")
