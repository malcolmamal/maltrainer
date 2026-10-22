"""Timestep sampling: the distribution, the shift and the quantisation to the scheduler's grid."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import train_zimage_lora as lora

from .world import Workspace


def _timesteps(ws: Workspace, size: int = 64, grid: int = 1000, **train: Any) -> torch.Tensor:
    trainer = lora.ZImageLoRATrainer(ws.lora_config(train=train))
    torch.manual_seed(3)
    return trainer._sample_timesteps(size, grid)


def test_uniform_timesteps_are_spread_over_the_unit_interval_on_the_grid(workspace: Workspace) -> None:
    steps = _timesteps(workspace, size=512)

    assert steps.shape == (512,) and steps.dtype == torch.float32
    assert steps.min() >= 0.0 and steps.max() < 1.0
    assert 0.4 < steps.mean() < 0.6
    assert torch.allclose(steps * 1000, (steps * 1000).round(), atol=1e-3)


@pytest.mark.parametrize("name", ["logit_normal", "logit-normal", "LogitNormal"])
def test_logit_normal_names_in_any_case_squeeze_zero_variance_onto_the_sigmoid_of_the_mean(
    workspace: Workspace, name: str
) -> None:
    steps = _timesteps(workspace, timestep_distribution=name, timestep_logit_mean=0.0, timestep_logit_std=0.0)

    assert torch.equal(steps, torch.full((64,), 0.5))


def test_the_logit_mean_moves_the_distribution(workspace: Workspace) -> None:
    low = _timesteps(workspace, size=256, timestep_distribution="logit_normal", timestep_logit_mean=-2.0)
    high = _timesteps(workspace, size=256, timestep_distribution="logit_normal", timestep_logit_mean=2.0)

    assert low.mean() < 0.3 and high.mean() > 0.7


def test_an_unknown_distribution_name_falls_back_to_uniform(workspace: Workspace) -> None:
    assert torch.equal(_timesteps(workspace, timestep_distribution="beta"), _timesteps(workspace))


@pytest.mark.parametrize(("shift", "expected"), [(0.5, 0.25), (1.0, 0.5), (1.5, 0.75), (4.0, 1.0)])
def test_the_shift_multiplies_the_timestep_and_clamps_at_one(
    workspace: Workspace, shift: float, expected: float
) -> None:
    steps = _timesteps(workspace, timestep_distribution="logit_normal", timestep_logit_std=0.0, timestep_shift=shift)

    assert torch.equal(steps, torch.full((64,), expected))


def test_the_grid_floors_a_timestep_to_a_multiple_of_one_over_its_size_and_zero_leaves_it_alone(
    workspace: Workspace,
) -> None:
    coarse = _timesteps(workspace, grid=4)
    raw = _timesteps(workspace, grid=0)

    assert set(coarse.tolist()) <= {0.0, 0.25, 0.5, 0.75}
    assert torch.equal(coarse, (raw * 4).floor() / 4)
    assert len(set(raw.tolist())) == 64
