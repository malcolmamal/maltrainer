"""What the LoRA trainer caches: image sizes and aspect, the prompts it encodes and the order it keeps."""

from __future__ import annotations

import logging

import pytest

from .fakes import Fakes
from .runs import cached_lora
from .world import SAMPLE, Workspace, write_image


def test_buckets_keep_each_images_aspect_on_a_sixteen_pixel_grid(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    write_image(workspace.data / "p" / "wide.png", "wide", size=(64, 32))
    write_image(workspace.data / "p" / "square.png", "square", size=(32, 32))

    with caplog.at_level(logging.INFO):
        trainer = cached_lora(workspace.lora_config())

    shapes = {
        key.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]: tuple(value.shape) for key, value in trainer.latent_cache.items()
    }
    assert shapes == {"square.png": (128, 2, 2), "wide.png": (128, 1, 2)}
    assert "Training sizes (w x h: images): 32x16: 1, 32x32: 1" in caplog.text


def test_square_mode_crops_every_image_to_the_resolution(workspace: Workspace) -> None:
    write_image(workspace.data / "p" / "wide.png", "wide", size=(64, 32))

    trainer = cached_lora(workspace.lora_config(dataset={"aspect_mode": "SQUARE"}))

    assert [tuple(value.shape) for value in trainer.latent_cache.values()] == [(128, 2, 2)]


def test_samples_follow_the_sorted_path_order_whatever_the_encoding_order(workspace: Workspace) -> None:
    for name in ("b/z.png", "a/y.png", "b/a.png"):
        write_image(workspace.data / name, name)

    trainer = cached_lora(workspace.lora_config(dataset={"cache": True}))

    assert [key.replace("\\", "/").split("/data/")[1] for key, _ in trainer.samples] == [
        "a/y.png",
        "b/a.png",
        "b/z.png",
    ]


def test_caption_dropout_adds_the_empty_prompt_and_previews_add_theirs(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 1)

    trainer = cached_lora(workspace.lora_config(sample=SAMPLE, train={"caption_dropout": 0.1}))

    assert sorted(trainer.text_cache) == ["", "a photo of a person", "a photo of zed"]
    assert sorted(fakes.text.prompts) == sorted(trainer.text_cache)


def test_a_caption_shared_by_many_images_is_encoded_once(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 4, caption="the same words")

    trainer = cached_lora(workspace.lora_config())

    assert fakes.text.prompts == ["the same words"] and len(trainer.samples) == 4


def test_an_image_that_fails_to_encode_is_left_out_of_the_samples_and_warned_about(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 1)
    (workspace.data / "p" / "broken.png").write_bytes(b"not an image")

    with caplog.at_level(logging.WARNING):
        trainer = cached_lora(workspace.lora_config())

    assert len(trainer.samples) == 1 and "Skipping broken.png" in caplog.text
