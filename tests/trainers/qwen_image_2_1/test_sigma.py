"""Timestep sampling, and what one training step feeds the network and how it builds the loss."""

from __future__ import annotations

import math
import re
from typing import Any

import pytest
import torch

import train_qwen_image_2_1_lora as qwen
from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRANetwork

from .runs import prepared, progress_lines, rerun
from .world import Workspace


def _trainer(ws: Workspace, **train: Any) -> qwen.QwenImage21LoRATrainer:
    return qwen.QwenImage21LoRATrainer(ws.config(train=train))


def _seeded(draw: Any) -> torch.Tensor:
    torch.manual_seed(3)
    return draw()


@pytest.mark.parametrize("name", ["shift", "sigmoid", "logit_normal", "SHIFT"])
def test_the_sigmoid_family_draws_a_logit_normal_scaled_by_sigmoid_scale(workspace: Workspace, name: str) -> None:
    trainer = _trainer(workspace, timestep_distribution=name, sigmoid_scale=2.0)

    drawn = _seeded(lambda: trainer._sample_timesteps(6))

    torch.testing.assert_close(drawn, _seeded(lambda: torch.sigmoid(torch.randn(6) * 2.0)))


@pytest.mark.parametrize("train", [{}, {"timestep_distribution": "shift"}])
def test_the_default_schedule_is_the_shifted_logit_normal(workspace: Workspace, train: dict[str, Any]) -> None:
    drawn = _seeded(lambda: _trainer(workspace, **train)._sample_timesteps(4))

    torch.testing.assert_close(drawn, _seeded(lambda: torch.sigmoid(torch.randn(4))))


@pytest.mark.parametrize("name", ["uniform", "lownoise", "anything-else"])
def test_every_other_distribution_draws_uniform_timesteps(workspace: Workspace, name: str) -> None:
    drawn = _seeded(lambda: _trainer(workspace, timestep_distribution=name)._sample_timesteps(6))

    torch.testing.assert_close(drawn, _seeded(lambda: torch.rand(6)))


def test_sigmoid_timesteps_are_clamped_to_a_thousandth_from_either_end(workspace: Workspace) -> None:
    drawn = _trainer(workspace, sigmoid_scale=1e4)._sample_timesteps(64)

    assert set(drawn.tolist()) == set(torch.tensor([0.001, 0.999]).tolist())


class Forwards:
    """Every call of the LoRA network: its keyword arguments and its output, recorded around the real forward."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[dict[str, Any], torch.Tensor]] = []
        original = QwenImage21LoRANetwork.forward

        def spy(network: QwenImage21LoRANetwork, **kwargs: Any) -> torch.Tensor:
            output = original(network, **kwargs)
            self.calls.append((kwargs, output.detach()))
            return output

        monkeypatch.setattr(QwenImage21LoRANetwork, "forward", spy)


def _logged_loss(text: str) -> float:
    match = re.search(r"Loss: ([\d.]+)", text)
    assert match is not None
    return float(match.group(1))


def _fixed_timesteps(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    monkeypatch.setattr(
        qwen.QwenImage21LoRATrainer, "_sample_timesteps", lambda self, n: torch.full((n,), value, dtype=self.dtype)
    )


def _shifted(raw: float, tokens: int) -> float:
    slope = (0.9 - 0.5) / (8192 - 256)
    shift = math.exp(0.5 + slope * (tokens - 256))
    return shift * raw / (1.0 + (shift - 1.0) * raw)


@pytest.mark.parametrize(("resolution", "tokens"), [(32, 4), (64, 16)])
def test_the_timestep_is_the_raw_draw_shifted_by_the_dynamic_shift_of_the_token_count(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, resolution: int, tokens: int
) -> None:
    workspace.add_images(1)
    forwards = Forwards(monkeypatch)
    _fixed_timesteps(monkeypatch, 0.5)

    prepared(workspace.config(dataset={"resolution": resolution}, train={"max_steps": 1})).train()

    kwargs, _ = forwards.calls[0]
    assert kwargs["timestep"].tolist() == pytest.approx([_shifted(0.5, tokens)])
    assert (kwargs["latent_h"], kwargs["latent_w"]) == (resolution // 16, resolution // 16)


def test_the_network_sees_the_flow_matching_mix_and_the_loss_is_the_mse_to_noise_minus_latent(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(1)
    forwards = Forwards(monkeypatch)
    _fixed_timesteps(monkeypatch, 0.5)
    trainer = prepared(workspace.config(train={"max_steps": 1}))
    trainer.train()

    kwargs, prediction = forwards.calls[0]
    sigma = _shifted(0.5, 4)
    x0 = next(iter(trainer.latent_cache.values())).permute(0, 2, 3, 1).reshape(1, 4, 4)
    noise = (kwargs["hidden_states"] - (1.0 - sigma) * x0) / sigma
    expected = torch.nn.functional.mse_loss(prediction, noise - x0)
    logged = _logged_loss(progress_lines(workspace)[0])
    assert kwargs["hidden_states"].shape == (1, 4, 4) and noise.std() > 0.1
    assert logged == pytest.approx(expected.item(), abs=1e-5)
    assert torch.equal(kwargs["encoder_hidden_states"], trainer.text_cache["a photo of zed"])


def test_loss_weight_strength_scales_the_loss_of_an_otherwise_identical_run(workspace: Workspace) -> None:
    workspace.add_images(1)
    losses = []
    for index, strength in enumerate((1.0, 3.0)):
        out = workspace.root / f"out{index}"
        rerun(workspace.config(train={"max_steps": 1, "loss_weight_strength": strength}, output={"path": str(out)}))
        losses.append(_logged_loss((out / "training.log").read_text(encoding="utf-8")))

    assert losses[1] == pytest.approx(3.0 * losses[0], rel=1e-4)


def test_a_batch_of_two_stacks_latents_and_prompts_and_gives_every_item_its_own_timestep(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    forwards = Forwards(monkeypatch)

    prepared(workspace.config(train={"batch_size": 2, "max_steps": 1})).train()

    kwargs, prediction = forwards.calls[0]
    assert kwargs["hidden_states"].shape == (2, 4, 4) and kwargs["encoder_hidden_states"].shape == (2, 4, 6)
    assert kwargs["timestep"].shape == (2,) and prediction.shape == (2, 4, 4)


def test_a_bfloat16_run_feeds_the_network_bfloat16_inputs_and_still_trains_float32_adapters(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(1)
    forwards = Forwards(monkeypatch)

    trainer = prepared(workspace.config(dtype="bfloat16", train={"max_steps": 1}))
    trainer.train()

    kwargs, _ = forwards.calls[0]
    assert kwargs["hidden_states"].dtype == kwargs["encoder_hidden_states"].dtype == torch.bfloat16
    assert {p.dtype for p in trainer.lora_network.trainable_parameters()} == {torch.float32}
