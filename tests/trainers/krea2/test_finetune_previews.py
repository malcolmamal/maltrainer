"""Previews while a Krea 2 finetune trains: the sample: section rendered with the weights in training by the real
sampler (the VAE decoder is a stand-in), the gallery on key g, and prompts added mid-run encoded on the CPU."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import pytest
import torch
import yaml
from PIL import Image

from maltrainer.trainers.krea2 import finetune_loop
from maltrainer.training import loop

from .fakes import Fakes, ScriptedKeys, script_keys
from .runs import final_weights, prepared_finetune, run_finetune
from .world import Workspace, same_values

SAMPLE: dict[str, Any] = {
    "every": 2,
    "prompts": ["a photo of zed smiling", {"prompt": "a photo of a person", "enabled": False}],
    "negative": "blurry",
    "steps": 2,
    "cfg": 3.0,
    "width": 32,
    "height": 48,
}


def _samples(ws: Workspace) -> list[str]:
    return sorted(path.name for path in (ws.out / "samples").glob("*.png"))


def test_previews_render_before_the_first_step_every_n_steps_and_at_the_last(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images(2)

    run_finetune(workspace.finetune_config(sample=SAMPLE))

    assert _samples(workspace) == [
        "step000000.png", "step000000_00.png", "step000002.png", "step000002_00.png", "step000003.png",
        "step000003_00.png",
    ]  # fmt: skip
    with Image.open(workspace.out / "samples" / "step000002_00.png") as image:
        assert image.size == (32, 48)
    assert fakes.decoder.decoded == [(1, 4, 6, 4)] * 3  # three renders of a 48 x 32 image: a 6 x 4 latent
    assert fakes.decoder_loads == [torch.bfloat16]  # loaded once, in the training dtype


def test_every_preview_prompt_and_the_negative_are_encoded_with_the_dataset(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images(1)

    trainer = prepared_finetune(workspace.finetune_config(sample=SAMPLE))

    assert {"a photo of zed smiling", "a photo of a person", "blurry"} <= set(trainer.text_cache)
    assert fakes.cpu_text_loads == []


def test_a_resumed_run_does_not_render_the_base_model_again(workspace: Workspace) -> None:
    workspace.add_images(2)
    run_finetune(workspace.finetune_config(train={"max_steps": 1}))

    state = str(workspace.out / "ft_state")
    run_finetune(workspace.finetune_config(sample=SAMPLE, train={"max_steps": 2, "resume_from": state}))

    assert _samples(workspace) == ["step000002.png", "step000002_00.png"]


def test_previews_leave_the_training_unchanged(workspace: Workspace) -> None:
    workspace.add_images(2)

    run_finetune(workspace.finetune_config(output={"path": str(workspace.root / "plain")}))
    run_finetune(workspace.finetune_config(sample=SAMPLE, output={"path": str(workspace.root / "previews")}))

    assert same_values(final_weights(workspace.root / "plain"), final_weights(workspace.root / "previews"))


@pytest.fixture
def press(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[list[str]]], list[ScriptedKeys]]:
    def pressing(script: list[list[str]]) -> list[ScriptedKeys]:
        return script_keys(monkeypatch, [finetune_loop, loop], script)

    return pressing


def test_g_renders_the_gallery_of_the_prior_classes(workspace: Workspace, press: Callable[..., Any]) -> None:
    workspace.add_images(2)
    press([[], ["g"]])
    sample = {**SAMPLE, "every": 0, "gallery": {"count": 1, "prompts": {"person": ["a photo of {trigger}"]}}}

    trainer = run_finetune(workspace.finetune_config(sample=sample, prior={"weight": 1.0, "class_word": "person"}))

    gallery = workspace.out / "samples" / "gallery"
    assert sorted(path.name for path in gallery.iterdir()) == ["step000002.png", "step000002.txt", "step000002_zed.png"]
    assert "a photo of zed" in (gallery / "step000002.txt").read_text(encoding="utf-8")
    assert trainer.transformer.training  # back to training after the render


def test_without_people_there_is_no_gallery_key(workspace: Workspace, press: Callable[..., Any]) -> None:
    workspace.add_images(1)
    made = press([])

    run_finetune(workspace.finetune_config(sample={**SAMPLE, "gallery": False}))

    assert made[0].gallery is None and made[0].menu is not None


def test_a_prompt_added_mid_run_is_encoded_on_the_cpu_by_the_krea2_text_encoder(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images(1)
    trainer = prepared_finetune(workspace.finetune_config(sample=SAMPLE))

    trainer.previews.update({**SAMPLE, "prompts": [*SAMPLE["prompts"], "a photo of zed at the beach"]})
    trainer._encode_new_prompts()
    assert trainer._encoding is not None
    trainer._encoding.join(timeout=60)

    assert fakes.cpu_text_loads == [torch.device("cpu")]
    assert trainer.text_cache["a photo of zed at the beach"].device == torch.device("cpu")


@pytest.mark.parametrize(
    "width, height, hint", [(1080, 1480, "1072 or 1088 for 1080"), (1072, 1470, "1456 or 1472 for 1470")]
)
def test_a_preview_size_off_the_16_px_grid_is_refused_before_any_caching(
    workspace: Workspace, fakes: Fakes, width: int, height: int, hint: str
) -> None:
    workspace.add_images(1)
    with pytest.raises(
        ValueError, match=rf"sample\.width / height must be multiples of 16, got {width}x{height}: .*{hint}"
    ):
        run_finetune(workspace.finetune_config(sample={**SAMPLE, "width": width, "height": height}))
    assert fakes.vae_loads == 0  # refused at construction, not after the latents and the prior


def test_a_size_off_the_grid_saved_into_the_config_mid_run_keeps_the_last_good_one(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)
    config = workspace.root / "run.yaml"
    trainer = prepared_finetune(workspace.finetune_config(sample=SAMPLE))
    trainer.config_path = config
    config.write_text(yaml.safe_dump(workspace.finetune_config(sample=SAMPLE)), encoding="utf-8")
    trainer._watch_config()  # the first look only notes the file's time
    config.write_text(yaml.safe_dump(workspace.finetune_config(sample={**SAMPLE, "width": 1080})), encoding="utf-8")
    os.utime(config, (config.stat().st_atime, config.stat().st_mtime + 5))

    with caplog.at_level(logging.ERROR):
        trainer._watch_config()

    assert (trainer.previews.width, trainer.previews.height) == (32, 48)
    assert (
        "sample.width / height must be multiples of 16, got 1080x48" in caplog.text and "keeping 32x48" in caplog.text
    )
