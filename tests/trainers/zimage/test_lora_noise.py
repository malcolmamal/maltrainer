"""The noisy latents the transformer is fed: flow matching plus the optional noise and latent transforms."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from maltrainer.data.dataset import ZImageDataset

from .fakes import Fakes
from .runs import run_lora, watch_noise
from .tiny import build_vae
from .world import Workspace

SHIFT, SCALE = 0.1159, 0.3611


class Step:
    """The one training step of a one-image run: its latents, noise, timestep and what the transformer got."""

    def __init__(self, ws: Workspace, fakes: Fakes, noises: list[torch.Tensor]) -> None:
        pixels = ZImageDataset(str(ws.data), resolution=16)[0]["pixel_values"].unsqueeze(0)
        with torch.no_grad():
            mean = build_vae().encode(pixels).latent_dist.mode()
        self.latents = (mean - SHIFT) * SCALE
        self.noise = noises[0]
        self.t = 1.0 - fakes.calls[0].t.reshape(1, 1, 1, 1)
        self.fed = fakes.calls[0].x.squeeze(2)

    def blend(self, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (1 - self.t) * latents + self.t * noise


def _step(ws: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, **train: Any) -> Step:
    ws.add_images(1)
    noises = watch_noise(monkeypatch)
    run_lora(ws.lora_config(train=train))
    return Step(ws, fakes, noises)


def test_the_input_blends_the_scaled_latents_and_the_noise_by_the_timestep(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _step(workspace, fakes, monkeypatch)

    assert torch.allclose(step.fed, step.blend(step.latents, step.noise), atol=1e-5)


def test_the_latent_multiplier_scales_the_latents_but_not_the_noise(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _step(workspace, fakes, monkeypatch, latent_multiplier=2.0)

    assert torch.allclose(step.fed, step.blend(step.latents * 2, step.noise), atol=1e-5)


def test_the_noise_multiplier_scales_the_noise_but_not_the_latents(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _step(workspace, fakes, monkeypatch, noise_multiplier=3.0)

    assert torch.allclose(step.fed, step.blend(step.latents, step.noise * 3), atol=1e-5)


def test_the_dynamic_noise_offset_adds_half_the_latents_channel_means_to_the_noise(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _step(workspace, fakes, monkeypatch, dynamic_noise_offset=True)

    offset = step.latents.mean(dim=(2, 3), keepdim=True) / 2
    assert torch.allclose(step.fed, step.blend(step.latents, step.noise + offset), atol=1e-5)


def test_the_noisy_latent_multiplier_scales_the_blend(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _step(workspace, fakes, monkeypatch, noisy_latent_multiplier=0.5)

    assert torch.allclose(step.fed, step.blend(step.latents, step.noise) * 0.5, atol=1e-5)


def test_the_random_noise_multiplier_scales_each_channel_of_the_noise_by_its_own_factor(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _step(workspace, fakes, monkeypatch, random_noise_multiplier=0.5)

    noise = (step.fed - (1 - step.t) * step.latents) / step.t
    factor = noise / step.noise
    assert torch.allclose(factor, factor[..., :1, :1].expand_as(factor), atol=1e-3)
    assert not torch.allclose(factor, torch.ones_like(factor), atol=1e-2)


def test_the_random_noise_shift_adds_one_constant_per_channel_to_the_noise(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    step = _step(workspace, fakes, monkeypatch, random_noise_shift=0.5)

    shift = (step.fed - (1 - step.t) * step.latents) / step.t - step.noise
    assert torch.allclose(shift, shift[..., :1, :1].expand_as(shift), atol=1e-4)
    assert shift.abs().max() > 1e-2
