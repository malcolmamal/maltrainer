"""The flow-matching timestep: a uniform or logit-normal draw pushed through FLUX.3's fixed shift, then clamped."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from train_flux3_klein_lora import FLUX3_SHIFT, Flux3KleinLoRATrainer

from .world import Workspace


def _sigma(ws: Workspace, **train: Any) -> float:
    trainer = Flux3KleinLoRATrainer(ws.config(train=train))
    return float(trainer._sample_sigma())


def _shifted(base: float, shift: float) -> float:
    return shift * base / (1.0 + (shift - 1.0) * base)


def test_the_default_is_a_uniform_draw_through_the_fixed_6_93_shift(workspace: Workspace) -> None:
    torch.manual_seed(5)
    base = float(torch.rand(1))
    torch.manual_seed(5)

    assert _sigma(workspace) == pytest.approx(_shifted(base, FLUX3_SHIFT))


@pytest.mark.parametrize("name", ["logit_normal", "logit-normal", "sigmoid", "Logit_Normal"])
def test_every_logit_normal_spelling_draws_sigmoid_of_a_normal(workspace: Workspace, name: str) -> None:
    torch.manual_seed(5)
    base = float(torch.sigmoid(torch.randn(1) * 2.0 + 0.5))
    torch.manual_seed(5)

    sigma = _sigma(workspace, timestep_distribution=name, timestep_logit_mean=0.5, timestep_logit_std=2.0)

    assert sigma == pytest.approx(_shifted(base, FLUX3_SHIFT))


def test_a_zero_spread_logit_normal_is_deterministic_and_the_shift_is_configurable(workspace: Workspace) -> None:
    sigma = _sigma(workspace, timestep_distribution="logit_normal", timestep_logit_std=0.0, timestep_shift=3.0)

    assert sigma == pytest.approx(0.75)
    assert _sigma(workspace, timestep_distribution="logit_normal", timestep_logit_std=0.0, timestep_shift=1.0) == 0.5


@pytest.mark.parametrize(("limits", "expected"), [({"min_sigma": 0.95}, 0.95), ({"max_sigma": 0.6}, 0.6)])
def test_sigma_is_clamped_to_the_configured_range(
    workspace: Workspace, limits: dict[str, float], expected: float
) -> None:
    sigma = _sigma(workspace, timestep_distribution="logit_normal", timestep_logit_std=0.0, **limits)

    assert sigma == pytest.approx(expected)


def test_an_unknown_distribution_is_refused(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match=r"train\.timestep_distribution must be uniform or logit_normal, got 'beta'"):
        _sigma(workspace, timestep_distribution="beta")
