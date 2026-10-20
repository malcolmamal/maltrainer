"""Timestep sampling, the sigma shift each schedule resolves to, and the contrastive guidance target."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

import train_minimax_h3_lora as h3

from .world import Workspace


def _trainer(ws: Workspace, **train: Any) -> h3.MiniMaxH3LoRATrainer:
    return h3.MiniMaxH3LoRATrainer(ws.config(train=train))


def _seeded(draw: Any) -> torch.Tensor:
    torch.manual_seed(3)
    return draw()


@pytest.mark.parametrize("name", ["krea2_shift", "krea2", "sigmoid", "KREA2_SHIFT"])
def test_the_sigmoid_family_draws_a_logit_normal_scaled_by_sigmoid_scale(workspace: Workspace, name: str) -> None:
    trainer = _trainer(workspace, timestep_distribution=name, sigmoid_scale=2.0)

    drawn = _seeded(lambda: trainer._sample_timesteps(6))

    expected = _seeded(lambda: torch.sigmoid(torch.randn(6) * 2.0))
    assert drawn.dtype == torch.float32
    torch.testing.assert_close(drawn, expected)


@pytest.mark.parametrize("name", ["logit_normal", "logit-normal", "logitnormal"])
def test_logit_normal_uses_the_configured_mean_and_std(workspace: Workspace, name: str) -> None:
    trainer = _trainer(workspace, timestep_distribution=name, timestep_logit_mean=-1.0, timestep_logit_std=0.5)

    drawn = _seeded(lambda: trainer._sample_timesteps(6))

    torch.testing.assert_close(drawn, _seeded(lambda: torch.sigmoid(torch.randn(6) * 0.5 - 1.0)))


@pytest.mark.parametrize("name", ["lownoise", "uniform", "anything-else"])
def test_every_other_distribution_draws_uniform_timesteps(workspace: Workspace, name: str) -> None:
    trainer = _trainer(workspace, timestep_distribution=name)

    drawn = _seeded(lambda: trainer._sample_timesteps(6))

    torch.testing.assert_close(drawn, _seeded(lambda: torch.rand(6)))


@pytest.mark.parametrize("name", ["krea2_shift", "logit_normal", "uniform"])
def test_timesteps_are_clamped_to_a_thousandth_from_either_end(workspace: Workspace, name: str) -> None:
    trainer = _trainer(workspace, timestep_distribution=name, sigmoid_scale=1e4, timestep_logit_std=1e4)

    drawn = trainer._sample_timesteps(64)

    assert drawn.min() >= 0.001 and drawn.max() <= 0.999
    if name != "uniform":
        assert set(drawn.tolist()) <= set(torch.tensor([0.001, 0.999]).tolist())


@pytest.mark.parametrize(
    ("train", "latent", "expected"),
    [
        ({}, (32, 32), math.exp(0.5)),  # 256 packed tokens: the low anchor
        ({"timestep_distribution": "krea2"}, (160, 160), math.exp(1.15)),  # 6400 tokens: the high anchor
        ({}, (64, 64), math.exp(0.5 + (0.65 / 6144) * (1024 - 256))),
        ({"timestep_distribution": "sigmoid"}, (64, 64), 1.0),
        ({"timestep_distribution": "lownoise"}, (64, 64), 0.4 / 0.6),
        ({"timestep_distribution": "lownoise", "lownoise_pct": 25}, (64, 64), 3.0),
        ({"timestep_distribution": "uniform"}, (64, 64), 12.0),
        ({"timestep_distribution": "logit_normal", "timestep_shift": 5.0}, (64, 64), 5.0),
    ],
)
def test_the_resolved_flow_shift_follows_the_schedule_and_the_packed_token_count(
    workspace: Workspace, train: dict[str, Any], latent: tuple[int, int], expected: float
) -> None:
    assert _trainer(workspace, **train)._resolve_flow_shift(*latent) == pytest.approx(expected)


@pytest.mark.parametrize("pct", [0, 100, -5, 130])
def test_lownoise_pct_must_lie_strictly_between_zero_and_a_hundred(workspace: Workspace, pct: int) -> None:
    trainer = _trainer(workspace, timestep_distribution="lownoise", lownoise_pct=pct)

    with pytest.raises(ValueError, match=r"train\.lownoise_pct must be between 0 and 100 \(exclusive\)"):
        trainer._resolve_flow_shift(8, 8)


def _guided(**options: Any) -> torch.Tensor:
    target = torch.tensor([[[2.0], [4.0]]])
    uncond = torch.tensor([[[1.0], [1.0]]])
    arguments: dict[str, Any] = {"sigma": torch.tensor([0.5]), "scale": 3.0, "schedule": "scale", "cfg_zero": False}
    arguments.update(options)
    return h3.MiniMaxH3LoRATrainer._guided_target(arguments.pop("target", target), uncond, **arguments)


def test_the_guided_target_extrapolates_from_the_unconditional_prediction() -> None:
    assert _guided().flatten().tolist() == [4.0, 10.0]  # 1 + 3 * (t - 1)


def test_the_sigma_schedule_decays_the_scale_toward_one_at_low_noise() -> None:
    assert _guided(schedule="sigma").flatten().tolist() == [3.0, 7.0]  # scale 1 + 2 * 0.5 = 2
    assert _guided(schedule="sigma", sigma=torch.tensor([0.0])).flatten().tolist() == [2.0, 4.0]  # the plain target


def test_each_batch_item_uses_its_own_sigma() -> None:
    target = torch.tensor([[[2.0]], [[2.0]]])

    guided = h3.MiniMaxH3LoRATrainer._guided_target(
        target, torch.zeros(2, 1, 1), torch.tensor([0.0, 1.0]), 3.0, "sigma", False
    )

    assert guided.flatten().tolist() == [2.0, 6.0]


def test_cfg_zero_first_rescales_the_unconditional_onto_the_conditional_direction() -> None:
    # u = (1, 1), t = (2, 4): dot / |u|^2 = 3, so u becomes (3, 3) and the target 3 + 3 * (t - 3).
    guided = _guided(cfg_zero=True)

    torch.testing.assert_close(guided.flatten(), torch.tensor([0.0, 6.0]))


def test_the_guided_target_keeps_the_target_dtype_and_carries_no_graph() -> None:
    target = torch.tensor([[[2.0], [4.0]]], dtype=torch.bfloat16, requires_grad=True)

    guided = _guided(target=target)

    assert guided.dtype == torch.bfloat16 and not guided.requires_grad
