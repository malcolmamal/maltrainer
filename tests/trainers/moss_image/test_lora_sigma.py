"""The noise level drawn for each step: the report's logit-normal by default, uniform, mix, shift, grid and clamp."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import train_moss_image_lora as lora

from .world import Workspace

FLAT: dict[str, Any] = {"timestep_uniform_mix": 0.0, "timestep_grid": 0, "min_sigma": 0.0, "max_sigma": 1.0}


def _draws(ws: Workspace, count: int = 1, **train: Any) -> list[float]:
    trainer = lora.MOSSImageLoRATrainer(ws.lora_config(train={**FLAT, **train}))
    torch.manual_seed(5)
    return [float(trainer._sample_sigma()) for _ in range(count)]


def _uniform(count: int = 1) -> list[float]:
    torch.manual_seed(5)
    return [float(torch.rand(1)) for _ in range(count)]


def test_uniform_draws_one_value_in_the_unit_interval(workspace: Workspace) -> None:
    assert _draws(workspace, timestep_distribution="uniform") == _uniform()


@pytest.mark.parametrize("shift", [0.5, 3.0, 5.0])
def test_the_shift_bends_a_uniform_draw_toward_one_for_shifts_above_one(workspace: Workspace, shift: float) -> None:
    u = _uniform()[0]

    drawn = _draws(workspace, timestep_distribution="uniform", timestep_shift=shift)[0]

    assert drawn == pytest.approx(shift * u / (1 + (shift - 1) * u), rel=1e-6)


def test_the_grid_rounds_up_onto_one_over_n_steps(workspace: Workspace) -> None:
    u = _uniform()[0]

    drawn = _draws(workspace, timestep_distribution="uniform", timestep_grid=10)[0]

    assert drawn == pytest.approx(torch.ceil(torch.tensor(u * 10)).item() / 10)


def test_the_default_grid_is_the_thousand_training_sigmas(workspace: Workspace) -> None:
    trainer = lora.MOSSImageLoRATrainer(workspace.lora_config())

    drawn = [float(trainer._sample_sigma()) * 1000 for _ in range(200)]

    assert all(value == pytest.approx(round(value)) and 1 <= round(value) <= 1000 for value in drawn)


def test_logit_normal_without_a_cutoff_is_the_sigmoid_of_a_normal_draw(workspace: Workspace) -> None:
    torch.manual_seed(5)
    expected = torch.sigmoid(torch.randn(1) * 0.5 + 0.25).item()

    drawn = _draws(workspace, timestep_logit_mean=0.25, timestep_logit_std=0.5, timestep_cutoff=1.0)[0]

    assert drawn == pytest.approx(expected, rel=1e-6)


def test_logit_normal_with_a_cutoff_is_truncated_by_inverse_cdf_and_rescaled_onto_the_unit_interval(
    workspace: Workspace,
) -> None:
    mean, std, cutoff = 1.09, 0.95, 0.95
    top = torch.special.ndtr(torch.tensor((torch.logit(torch.tensor(cutoff)).item() - mean) / std))
    torch.manual_seed(5)
    expected = (torch.sigmoid(mean + std * torch.special.ndtri(torch.rand(1) * top)) / cutoff).item()

    assert _draws(workspace)[0] == pytest.approx(expected, rel=1e-5)
    assert max(_draws(workspace, 300)) <= 1.0


@pytest.mark.parametrize("name", ["logit_normal", "logit-normal", "Sigmoid", "LOGIT_NORMAL"])
def test_the_logit_normal_spellings_agree(workspace: Workspace, name: str) -> None:
    assert _draws(workspace, 3, timestep_distribution=name) == _draws(
        workspace, 3, timestep_distribution="logit_normal"
    )


def test_a_uniform_mix_of_one_replaces_every_draw_with_a_fresh_uniform_one(workspace: Workspace) -> None:
    torch.manual_seed(5)
    _first, _check, replacement = torch.rand(1), torch.rand(()), torch.rand(1)

    drawn = _draws(workspace, timestep_distribution="uniform", timestep_uniform_mix=1.0)[0]

    assert drawn == pytest.approx(replacement.item())


def test_draws_are_clamped_to_min_and_max_sigma(workspace: Workspace) -> None:
    drawn = _draws(workspace, 100, timestep_distribution="uniform", min_sigma=0.4, max_sigma=0.6)

    assert min(drawn) == pytest.approx(0.4) and max(drawn) == pytest.approx(0.6)
    assert all(0.4 - 1e-6 <= value <= 0.6 + 1e-6 for value in drawn)


def test_an_unknown_distribution_is_refused(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match=r"train\.timestep_distribution must be logit_normal or uniform, got 'beta'"):
        _draws(workspace, timestep_distribution="Beta")
