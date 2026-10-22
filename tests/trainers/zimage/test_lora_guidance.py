"""Differential guidance: a second, unconditional pass whose difference to the first is amplified."""

from __future__ import annotations

from typing import Any

import pytest

import train_zimage_lora as lora

from .fakes import Fakes
from .runs import first_step, watch_losses
from .world import Workspace

GUIDED = {"do_differential_guidance": True}


def _guided(ws: Workspace, fakes: Fakes, mp: pytest.MonkeyPatch, scale: float, **train: Any) -> float:
    pinned = {"timestep_distribution": "logit_normal", "timestep_logit_std": 0.0, "differential_guidance_scale": scale}
    return first_step(ws, fakes, mp, train={**GUIDED, **pinned, **train})[0]


def test_each_step_runs_the_transformer_twice_with_the_conditional_then_the_unconditional_text(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_step(workspace, fakes, monkeypatch, train=GUIDED)

    assert len(fakes.calls) == 2
    assert fakes.calls[0].x.equal(fakes.calls[1].x) and fakes.calls[0].t.equal(fakes.calls[1].t)
    assert fakes.prompts() == ["<|user|> zed a photo of person <|assistant|> think", "<|user|> <|assistant|>"]


def test_a_custom_unconditional_prompt_is_the_second_pass_text(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_step(workspace, fakes, monkeypatch, train={**GUIDED, "unconditional_prompt": "a person"})

    assert fakes.prompts()[1] == "<|user|> a person <|assistant|>"


def test_the_guided_prediction_is_the_unconditional_one_plus_scale_times_the_difference(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    at0, at1, at2 = (_guided(workspace, fakes, monkeypatch, s) for s in (0.0, 1.0, 2.0))

    curvature = (at2 - 2 * at1 + at0) / 2
    slope = at1 - at0 - curvature
    assert _guided(workspace, fakes, monkeypatch, 3.0) == pytest.approx(at0 + 3 * slope + 9 * curvature, rel=1e-3)


def test_a_scale_of_one_leaves_the_conditional_prediction_as_it_is(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = first_step(workspace, fakes, monkeypatch)[0]

    unit = {**GUIDED, "differential_guidance_scale": 1.0}
    assert first_step(workspace, fakes, monkeypatch, train=unit)[0] == pytest.approx(plain, rel=1e-4)


@pytest.mark.parametrize("output", ["tuple", "tensor", "plain"])
def test_the_unconditional_output_shapes_are_handled_like_the_conditional_ones(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, output: str
) -> None:
    expected = first_step(workspace, fakes, monkeypatch, train=GUIDED)[0]
    fakes.output = output

    assert first_step(workspace, fakes, monkeypatch, train=GUIDED)[0] == pytest.approx(expected, rel=1e-5)


def test_a_transposed_unconditional_output_is_put_back_and_a_mask_goes_with_it(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    fakes.masked = True
    losses = []
    for output in ("plain", "swapped"):
        fakes.output = output
        trainer = lora.ZImageLoRATrainer(workspace.lora_config(train={**GUIDED, "batch_size": 2}))
        losses.append(watch_losses(trainer, monkeypatch))
        trainer.run()

    assert losses[0] == pytest.approx(losses[1], rel=1e-5)
    assert fakes.calls[1].cap_mask is not None and fakes.calls[1].cap_mask.shape[0] == 2


@pytest.mark.parametrize(
    ("chat", "first", "second"),
    [
        ("none", "zed a photo of person", "a person"),
        ("plain", "<|user|> zed a photo of person <|assistant|>", "<|user|> a person <|assistant|>"),
    ],
)
def test_a_tokenizer_without_a_chat_template_or_its_thinking_flag_gets_the_text_it_can_take(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, chat: str, first: str, second: str
) -> None:
    fakes.chat = chat

    first_step(workspace, fakes, monkeypatch, train={**GUIDED, "unconditional_prompt": "a person"})

    assert fakes.prompts() == [first, second]
