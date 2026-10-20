"""The flow-matching timestep: a draw pushed through Krea 2's resolution-dependent shift (or a fixed one), clamped."""

from __future__ import annotations

import logging
import math
from typing import Any

import pytest
import torch

from train_krea2_lora import Krea2LoRATrainer, krea2_shift

from .runs import run_lora
from .world import Workspace

TOKENS = 100


def _draw(ws: Workspace, **train: Any) -> tuple[float, float]:
    trainer = Krea2LoRATrainer(ws.lora_config(train=train))
    sigma, shift = trainer._sample_sigma(TOKENS)
    return float(sigma), shift


def _shifted(base: float, shift: float) -> float:
    return shift * base / (1.0 + (shift - 1.0) * base)


@pytest.mark.parametrize(("tokens", "mu"), [(256, 0.5), (6400, 1.15), (3328, 0.825)])
def test_the_shift_grows_with_the_image_tokens_from_mu_half_at_256_to_1_15_at_6400(tokens: int, mu: float) -> None:
    assert krea2_shift(tokens) == pytest.approx(math.exp(mu))


def test_the_default_is_a_logit_normal_draw_through_the_shift_of_the_image_size(workspace: Workspace) -> None:
    torch.manual_seed(5)
    base = float(torch.sigmoid(torch.randn(1) * 2.0 + 0.5))
    torch.manual_seed(5)

    sigma, shift = _draw(workspace, timestep_logit_mean=0.5, timestep_logit_std=2.0)

    assert shift == pytest.approx(krea2_shift(TOKENS))
    assert sigma == pytest.approx(_shifted(base, shift))


@pytest.mark.parametrize(("name", "shift"), [("logit_normal", 3.0), ("logit-normal", 3.0), ("sigmoid", 1.0)])
def test_the_plain_logit_normal_names_use_the_configured_shift_and_sigmoid_never_shifts(
    workspace: Workspace, name: str, shift: float
) -> None:
    sigma, used = _draw(workspace, timestep_distribution=name, timestep_logit_std=0.0, timestep_shift=3.0)

    assert used == shift
    assert sigma == pytest.approx(_shifted(0.5, shift))


def test_uniform_draws_rand_and_defaults_to_no_shift(workspace: Workspace) -> None:
    torch.manual_seed(5)
    base = float(torch.rand(1))
    torch.manual_seed(5)

    sigma, shift = _draw(workspace, timestep_distribution="UNIFORM")

    assert (shift, sigma) == (1.0, pytest.approx(base))


@pytest.mark.parametrize(("limits", "expected"), [({"min_sigma": 0.95}, 0.95), ({"max_sigma": 0.3}, 0.3)])
def test_sigma_is_clamped_to_the_configured_range(
    workspace: Workspace, limits: dict[str, float], expected: float
) -> None:
    sigma, _ = _draw(workspace, timestep_logit_std=0.0, **limits)

    assert sigma == pytest.approx(expected)


def test_the_resolved_shift_is_recorded_and_logged_once_for_the_first_step(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)

    with caplog.at_level(logging.INFO):
        trainer = run_lora(workspace.lora_config())

    expected = round(krea2_shift(4), 6)
    assert trainer.run_metadata["resolved_sigma_shift"] == expected
    assert caplog.text.count("Timestep schedule: krea2_shift, sigma shift 1.605 at 4 image tokens") == 1
