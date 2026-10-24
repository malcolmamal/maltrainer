"""The training sigma: BRIA's token-dependent logit-normal by default, uniform or logit-normal on request."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from maltrainer.trainers.fibo2.timesteps import DISTRIBUTIONS, sample_sigma, token_shift

CPU = torch.device("cpu")


def _sigma(tokens: int = 1024, **train: Any) -> float:
    return float(sample_sigma(train, tokens, CPU))


@pytest.mark.parametrize(
    ("tokens", "shift"), [(256, 0.5), (4096, 1.15), (2176, 0.825), (1024, 0.5 + 0.65 * 768 / 3840)]
)
def test_the_mean_of_the_logit_normal_rises_linearly_with_the_token_count(tokens: int, shift: float) -> None:
    assert token_shift(tokens) == pytest.approx(shift)


def test_the_default_is_sigmoid_of_a_normal_around_the_token_shift() -> None:
    torch.manual_seed(5)
    mix = torch.rand(())  # the draw that decides between the tail and the body
    expected = float(torch.sigmoid(torch.randn(1) + token_shift(1024)))
    torch.manual_seed(5)

    assert mix.item() >= 0.1
    assert _sigma() == pytest.approx(expected)


def test_a_zero_mix_skips_the_uniform_tail_altogether() -> None:
    torch.manual_seed(5)
    expected = float(torch.sigmoid(torch.randn(1) + token_shift(1024)))
    torch.manual_seed(5)

    assert _sigma(timestep_uniform_mix=0.0) == pytest.approx(expected)


def test_a_full_mix_takes_the_uniform_draw_that_follows_the_mix_decision() -> None:
    torch.manual_seed(5)
    torch.rand(())  # the mix decision
    uniform = float(torch.rand(1))
    torch.manual_seed(5)

    assert _sigma(timestep_uniform_mix=1.0) == pytest.approx(uniform)


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
    with pytest.raises(ValueError, match=r"train\.timestep_distribution must be one of .*'bria'.*got 'beta'"):
        _sigma(timestep_distribution="beta")
    assert DISTRIBUTIONS == ("bria", "uniform", "logit_normal")
