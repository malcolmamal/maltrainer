"""Making the dataset and caching its latents and prompt embeddings before the transformer loads."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch

import train_qwen_image_2_1_lora as qwen

from .fakes import Fakes
from .runs import cached
from .world import Workspace, write_image


def test_the_dataset_has_one_item_per_image_and_repeat_and_batches_as_configured(workspace: Workspace) -> None:
    workspace.add_images(3)

    trainer = cached(workspace.config(dataset={"repeats": 2}, train={"batch_size": 2}))

    assert len(trainer.train_dataset) == 6 and len(trainer.train_dataloader) == 3
    assert trainer.train_dataloader.batch_size == 2 and trainer.train_dataloader.num_workers == 0


def test_every_unique_image_is_encoded_once_however_often_it_repeats(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images(2)

    trainer = cached(workspace.config(dataset={"repeats": 3}))

    assert len(trainer.latent_cache) == 2 and len(fakes.vae.encoded) == 2
    assert all(latent.shape == (1, 4, 2, 2) and latent.device.type == "cpu" for latent in trainer.latent_cache.values())


def test_the_vae_is_loaded_encoder_only_and_encodes_one_image_deterministically_at_the_run_dtype(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images(1)

    cached(workspace.config(dtype="bfloat16"))

    assert fakes.vae.encoded == [((1, 3, 32, 32), torch.bfloat16, True)]
    assert fakes.vae_options == [{"device": torch.device("cpu"), "dtype": torch.bfloat16, "include_decoder": False}]


def test_the_resolution_is_rounded_up_to_a_multiple_of_sixteen(workspace: Workspace, fakes: Fakes) -> None:
    write_image(workspace.data / "wide.png", "a wide one", size=(80, 40))

    trainer = cached(workspace.config(dataset={"resolution": 40}))

    assert fakes.vae.encoded[0][0] == (1, 3, 48, 48)
    assert next(iter(trainer.latent_cache.values())).shape == (1, 4, 3, 3)


def test_unique_captions_are_encoded_once_each_and_kept_on_the_cpu(workspace: Workspace, fakes: Fakes) -> None:
    write_image(workspace.data / "a.png", "zed in red")
    write_image(workspace.data / "b.png", "zed in red")
    write_image(workspace.data / "c.png", "zed dogs")

    trainer = cached(workspace.config())

    assert sorted(fakes.text.prompts) == ["zed dogs", "zed in red"]
    assert {k: tuple(v.shape) for k, v in trainer.text_cache.items()} == {
        "zed in red": (1, 3, 6),
        "zed dogs": (1, 2, 6),
    }
    assert {v.device.type for v in trainer.text_cache.values()} == {"cpu"}


def test_the_tokenizer_path_goes_to_the_text_encoder_loader(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images(1)

    cached(workspace.config(model={"tokenizer_path": "somewhere/tokenizer"}))

    assert fakes.text_options == [("somewhere/tokenizer", torch.device("cpu"))]


def test_the_default_caption_and_trigger_make_the_captions_of_uncaptioned_images(workspace: Workspace) -> None:
    workspace.add_images(1, caption=None)

    trainer = cached(workspace.config(dataset={"trigger": "zed", "default_caption": "a drawing"}))

    assert list(trainer.text_cache) == ["zed a drawing"]


@pytest.mark.parametrize("flag", ["do_guidance_loss", "do_differential_guidance"])
def test_guidance_loss_is_ignored_with_a_warning_and_encodes_only_the_captions(
    workspace: Workspace, flag: str, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)

    with caplog.at_level(logging.WARNING):
        trainer = cached(workspace.config(train={flag: True, "unconditional_prompt": "blurry"}))

    assert sorted(trainer.text_cache) == ["a photo of zed"]
    assert "do_guidance_loss is not implemented" in caplog.text


def test_an_unconditional_prompt_that_is_also_a_caption_is_encoded_once(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images(1)

    cached(workspace.config(train={"do_guidance_loss": True, "unconditional_prompt": "a photo of zed"}))

    assert fakes.text.prompts == ["a photo of zed"]


def test_the_empty_unconditional_prompt_is_encoded_once_even_when_a_caption_is_empty(workspace: Workspace) -> None:
    workspace.add_images(1, caption=None)

    trainer = cached(workspace.config(dataset={"trigger": "", "default_caption": ""}, train={"do_guidance_loss": True}))

    assert list(trainer.text_cache) == [""]


def test_models_load_vae_first_then_text_and_the_transformer_waits(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images(1)

    cached(workspace.config())

    assert fakes.loads == {"vae": 1, "text": 1, "transformer": 0}


@pytest.mark.parametrize(
    ("train", "expected"),
    [
        ({}, (2, 5, 10)),
        ({"num_epochs": 3}, (2, 3, 6)),
        ({"max_steps": 5, "num_epochs": 99}, (2, 3, 5)),
        ({"gradient_accumulation_steps": 2, "num_epochs": 2}, (1, 2, 2)),
        ({"max_steps": 0, "num_epochs": 0}, (2, 1, 2)),
        ({"max_steps": None, "num_epochs": None, "gradient_accumulation_steps": None}, (2, 1, 2)),
    ],
)
def test_the_plan_gives_steps_per_epoch_epochs_and_total_steps(
    train: dict[str, Any], expected: tuple[int, int, int]
) -> None:
    assert qwen.plan_training_steps(2, {"max_steps": 0, "num_epochs": 5, **train}) == expected


def test_an_empty_dataset_still_plans_one_step_per_epoch() -> None:
    assert qwen.plan_training_steps(0, {}) == (1, 10, 10)
