"""The loss: timestep weighting, the constant weight, gradient clipping and the transformer's output shapes."""

from __future__ import annotations

import math
from typing import Any

import pytest

import train_zimage_lora as lora

from .fakes import Fakes
from .runs import first_step, watch_losses
from .world import Workspace

AT = {0.0: -20.0, 0.5: 0.0, 1.0: 20.0}
"""The logit mean that makes every sampled timestep 0, 0.5 or 1 (with a standard deviation of 0)."""
AITK = {0.0: 0.905706524848938, 0.5: 1.4976568222045898, 1.0: 1.458650827407837}


def _loss(ws: Workspace, fakes: Fakes, mp: pytest.MonkeyPatch, t: float, **train: Any) -> float:
    pinned = {"timestep_distribution": "logit_normal", "timestep_logit_std": 0.0, "timestep_logit_mean": AT[t]}
    loss, drawn = first_step(ws, fakes, mp, train={**pinned, **train})
    assert drawn == pytest.approx(t, abs=1e-6)
    return loss


@pytest.mark.parametrize(
    ("name", "t", "weight"),
    [
        ("linear", 0.5, 1.0),
        ("linear", 1.0, 1.5),
        ("sqrt", 0.5, 0.5 + math.sqrt(0.5)),
        ("cosine", 1.0, 1.5),
        ("cosine", 0.0, 0.5),
        ("AITK", 0.0, AITK[0.0]),
        ("aitk", 0.5, AITK[0.5]),
        ("aitk", 1.0, AITK[1.0]),
        ("bogus", 0.5, 1.0),
    ],
)
def test_timestep_weighting_scales_the_per_sample_loss_by_its_formula(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, name: str, t: float, weight: float
) -> None:
    plain = _loss(workspace, fakes, monkeypatch, t)

    weighted = _loss(workspace, fakes, monkeypatch, t, timestep_weighting=name)

    assert weighted == pytest.approx(plain * weight, rel=1e-4)


@pytest.mark.parametrize(("name", "factor"), [("constant", 2.5), ("other", 1.0)])
def test_the_constant_loss_weight_multiplies_the_loss_by_its_strength(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, name: str, factor: float
) -> None:
    plain = _loss(workspace, fakes, monkeypatch, 0.5)

    scaled = _loss(workspace, fakes, monkeypatch, 0.5, loss_weight_fn=name, loss_weight_strength=2.5)

    assert scaled == pytest.approx(plain * factor, rel=1e-4)


@pytest.mark.parametrize("output", ["tuple", "tensor", "plain"])
def test_every_shape_the_transformer_may_return_gives_the_same_loss(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    expected = first_step(workspace, fakes, monkeypatch)[0]
    fakes.output = output

    assert first_step(workspace, fakes, monkeypatch)[0] == pytest.approx(expected, rel=1e-5)


def test_a_transposed_batch_and_channel_output_is_put_back_before_the_loss(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    losses = []
    for output in ("plain", "swapped"):
        fakes.output = output
        trainer = lora.ZImageLoRATrainer(workspace.lora_config(train={"batch_size": 2}))
        losses.append(watch_losses(trainer, monkeypatch))
        trainer.run()

    assert losses[0] == pytest.approx(losses[1], rel=1e-5)


def test_a_transformer_with_a_cap_mask_parameter_is_given_the_boolean_attention_mask(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = first_step(workspace, fakes, monkeypatch)[0]
    fakes.masked = True

    assert first_step(workspace, fakes, monkeypatch)[0] == pytest.approx(expected, rel=1e-5)
    mask = fakes.calls[0].cap_mask
    assert mask is not None and mask.dtype.is_floating_point is False
    assert mask.all() and mask.shape == fakes.calls[0].cap_feats.shape[:2]


def test_the_gradient_norm_is_clipped_to_max_grad_norm_with_one_as_the_default(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(1)
    limits: list[float] = []
    for config in ({}, {"max_grad_norm": 0.25}):
        trainer = lora.ZImageLoRATrainer(workspace.lora_config(train=config))
        real = trainer.accelerator.clip_grad_norm_

        def clip(parameters: Any, max_norm: float, real: Any = real) -> Any:
            limits.append(max_norm)
            return real(parameters, max_norm)

        monkeypatch.setattr(trainer.accelerator, "clip_grad_norm_", clip)
        trainer.run()

    assert limits == [1.0, 0.25]
