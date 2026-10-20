"""Previews during a LoRA run, and how they leave the training and the encoded prompts."""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from PIL import Image

from maltrainer.utils.ema import ParameterEMA

from .fakes import Fakes
from .runs import cached_lora, prepared_lora, run_lora, same_values
from .world import SAMPLE, Workspace


def _files(ws: Workspace) -> list[str]:
    return sorted(path.name for path in (ws.out / "samples").iterdir())


def _lora_state(trainer: Any) -> dict[str, torch.Tensor]:
    return trainer.accelerator.unwrap_model(trainer.lora_network).get_state_dict(None)


def test_previews_render_before_the_first_step_every_n_steps_and_at_the_end(workspace: Workspace) -> None:
    workspace.add_images("p", 2)

    run_lora(workspace.lora_config(sample=SAMPLE, train={"max_steps": 3}))

    assert _files(workspace) == [
        "prompts.txt",
        "step000000.png",
        "step000000_00.png",
        "step000002.png",
        "step000002_00.png",
        "step000003.png",
        "step000003_00.png",
    ]


def test_a_preview_is_an_rgb_image_of_the_requested_size(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 1)
    trainer = prepared_lora(workspace.lora_config(sample={**SAMPLE, "width": 48, "height": 32}))

    image = trainer._render_preview("a photo of zed", 3)

    assert (image.mode, image.size) == ("RGB", (48, 32))
    assert fakes.decoder_loads == 1 and isinstance(image, Image.Image)


def test_without_cfg_no_negative_prompt_is_encoded_or_needed(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    trainer = cached_lora(workspace.lora_config(sample={**SAMPLE, "cfg": 1.0}))

    assert "" not in trainer.text_cache and trainer._preview_ready("a photo of zed")
    assert sorted(trainer.text_cache) == ["a photo of a person", "a photo of zed"]


def test_previews_leave_the_training_result_bit_for_bit_unchanged_and_the_adapters_back_in_train_mode(
    workspace: Workspace,
) -> None:
    workspace.add_images("p", 2)
    ema = {"ema_decay": 0.5, "ema_warmup": False}
    quiet = run_lora(workspace.lora_config(output={"path": str(workspace.root / "quiet")}, train=ema))

    shown = run_lora(workspace.lora_config(sample={**SAMPLE, "every": 1}, train=ema))

    assert same_values(_lora_state(shown), _lora_state(quiet))
    assert shown.ema is not None and quiet.ema is not None
    assert all(torch.equal(a, b) for a, b in zip(shown.ema.shadow, quiet.ema.shadow, strict=True))
    assert shown.lora_network.training


def test_the_previews_of_a_lora_run_show_the_average_that_checkpoints_save(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    plain = prepared_lora(workspace.lora_config())
    averaged = prepared_lora(workspace.lora_config())
    averaged.ema = ParameterEMA(averaged.accelerator.unwrap_model(averaged.lora_network).trainable_parameters())

    [(label, context)] = averaged._preview_variants()
    [(plain_label, plain_context)] = plain._preview_variants()

    assert (label, plain_label) == ("", "")
    assert context == averaged.ema.applied and plain_context is contextlib.nullcontext


def test_a_second_process_renders_no_previews(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 1)
    trainer = prepared_lora(workspace.lora_config(sample=SAMPLE))
    trainer.is_main_process = False

    seconds = trainer._maybe_preview(0, 3)

    assert seconds == 0.0 and fakes.decoder_loads == 0 and not (workspace.out / "samples" / "step000000.png").exists()
