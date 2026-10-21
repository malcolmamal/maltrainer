"""The sigma pieces the trainers share, value by value."""

from __future__ import annotations

import pytest
import torch

from maltrainer.training.sigma import clamped, logit_normal, shifted

CPU = torch.device("cpu")


def test_logit_normal_defaults_to_a_standard_normal_before_the_sigmoid() -> None:
    torch.manual_seed(3)
    expected = torch.sigmoid(torch.randn(1))
    torch.manual_seed(3)

    assert torch.equal(logit_normal({}, CPU), expected)


def test_logit_normal_takes_its_mean_and_spread_from_the_config_then_from_the_arguments() -> None:
    torch.manual_seed(3)
    noise = torch.randn(1)
    from_config = {"timestep_logit_mean": 0.5, "timestep_logit_std": 2.0}

    torch.manual_seed(3)
    assert torch.equal(logit_normal(from_config, CPU), torch.sigmoid(noise * 2.0 + 0.5))
    torch.manual_seed(3)
    assert torch.equal(logit_normal({}, CPU, mean=-1.0, std=0.5), torch.sigmoid(noise * 0.5 - 1.0))
    torch.manual_seed(3)
    assert torch.equal(logit_normal({"timestep_logit_mean": 1.0}, CPU), torch.sigmoid(noise + 1.0))


def test_the_shift_maps_sigma_by_s_sigma_over_one_plus_s_minus_one_sigma() -> None:
    sigma = torch.tensor([0.0, 0.25, 0.5, 1.0])

    torch.testing.assert_close(shifted(sigma, 3.0), torch.tensor([0.0, 0.5, 0.75, 1.0]))
    assert torch.equal(shifted(sigma, 1.0), sigma)


def test_clamped_keeps_sigma_inside_the_configured_range_and_defaults_to_the_open_ends() -> None:
    sigma = torch.tensor([0.0, 0.5, 1.0])

    torch.testing.assert_close(clamped(sigma, {}), torch.tensor([0.001, 0.5, 0.999]))
    torch.testing.assert_close(clamped(sigma, {"min_sigma": 0.2, "max_sigma": 0.7}), torch.tensor([0.2, 0.5, 0.7]))
    torch.testing.assert_close(clamped(sigma, {}, max_sigma=0.8), torch.tensor([0.001, 0.5, 0.8]))
    assert clamped(sigma, {"max_sigma": 0.9}, max_sigma=0.8)[-1] == pytest.approx(0.9)
