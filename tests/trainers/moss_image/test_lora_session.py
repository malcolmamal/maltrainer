"""A LoRA session: loss options, precision, key commands, periodic saves and a second process."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from accelerate import Accelerator

import train_moss_image_lora as lora
from maltrainer.trainers.moss_image import finetune_loop

from .fakes import script_keys
from .runs import prepared_lora, progress_lines, run_lora
from .world import SAMPLE, Workspace


def test_loss_weight_strength_scales_the_loss_and_leaves_the_draws_alone(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    plain = prepared_lora(workspace.lora_config())
    strong = prepared_lora(workspace.lora_config(train={"loss_weight_strength": 3.0}))
    torch.manual_seed(1)
    base = plain._training_step(0).item()
    torch.manual_seed(1)

    assert strong._training_step(0).item() == pytest.approx(3.0 * base, rel=1e-5)


def test_caption_dropout_trains_on_the_empty_prompt_the_cfg_negative_uses(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images("p", 1)
    trainer = prepared_lora(workspace.lora_config(train={"caption_dropout": 1.0}))
    contexts: list[torch.Tensor] = []
    forward = trainer.transformer.forward

    def spy(x: torch.Tensor, sigma: torch.Tensor, context: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        contexts.append(context)
        return forward(x, sigma, context, **kwargs)

    monkeypatch.setattr(trainer.transformer, "forward", spy)
    trainer._training_step(0)

    assert torch.equal(contexts[0][0].float(), trainer.text_cache[""].to(trainer.dtype).float())


@pytest.mark.parametrize("mixed", ["no", "bf16"])
def test_a_run_works_with_and_without_mixed_precision(workspace: Workspace, mixed: str) -> None:
    workspace.add_images("p", 2)

    trainer = run_lora(workspace.lora_config(mixed_precision=mixed, train={"max_steps": 2}))

    assert (workspace.out / "moss_image_lora.safetensors").exists()
    assert str(trainer.accelerator.mixed_precision) == mixed


def test_s_saves_a_numbered_lora_and_q_saves_and_stops(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2)
    script_keys(monkeypatch, [lora, finetune_loop], [["s"], ["q"]])

    with caplog.at_level(logging.INFO):
        run_lora(workspace.lora_config(train={"max_steps": 5}))

    saved = sorted(path.name for path in workspace.out.glob("*.safetensors"))
    assert saved == ["moss_image_lora_step_1.safetensors", "moss_image_lora_step_2.safetensors"]
    assert "MOSS-Image LoRA training stopped by key at step 2 (saved as step 2)." in caplog.text


def test_save_every_writes_numbered_loras_before_the_final_one(workspace: Workspace) -> None:
    workspace.add_images("p", 2)

    run_lora(workspace.lora_config(train={"save_every": 2, "max_steps": 4}))

    assert sorted(path.name for path in workspace.out.glob("*.safetensors")) == [
        "moss_image_lora.safetensors",
        "moss_image_lora_step_2.safetensors",
    ]


def test_a_second_process_trains_without_saving_previewing_or_listening_for_keys(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Accelerator, "is_main_process", property(lambda _self: False))
    workspace.add_images("p", 2)

    trainer = run_lora(workspace.lora_config(sample=SAMPLE, train={"max_steps": 2}))

    assert not trainer.is_main_process and len(progress_lines(workspace)) == 2
    assert not list(workspace.out.glob("*.safetensors")) and not (workspace.out / "samples").exists()
