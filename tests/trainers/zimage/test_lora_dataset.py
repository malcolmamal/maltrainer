"""The dataset the LoRA trainer builds: captions, trigger word, repeats, cropping and batching."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from torchvision import transforms

import train_zimage_lora as lora

from .world import Workspace, write_image


def _dataset(ws: Workspace, **sections: Any) -> lora.ZImageLoRATrainer:
    trainer = lora.ZImageLoRATrainer(ws.lora_config(**sections))
    trainer.create_dataset()
    return trainer


def _captions(trainer: lora.ZImageLoRATrainer) -> dict[str, str]:
    return {
        path.stem: caption
        for path, caption in zip(trainer.train_dataset.image_paths, trainer.train_dataset.captions, strict=True)
    }


def _mixed(ws: Workspace) -> None:
    write_image(ws.data / "empty.png")
    write_image(ws.data / "named.png", "a photo of person")


def test_the_trigger_leads_each_caption_and_the_default_replaces_a_missing_one(workspace: Workspace) -> None:
    _mixed(workspace)

    trainer = _dataset(workspace)

    assert _captions(trainer) == {"empty": "zed a photo", "named": "zed a photo of person"}


def test_a_default_caption_that_holds_the_trigger_is_used_as_it_is(workspace: Workspace) -> None:
    _mixed(workspace)

    trainer = _dataset(workspace, dataset={"default_caption": "photo of zed"})

    assert _captions(trainer) == {"empty": "photo of zed", "named": "zed a photo of person"}


def test_without_a_trigger_the_default_caption_alone_fills_in_the_missing_ones(workspace: Workspace) -> None:
    _mixed(workspace)

    trainer = _dataset(workspace, dataset={"trigger": ""})

    assert _captions(trainer) == {"empty": "a photo", "named": "a photo of person"}


def test_empty_captions_without_a_trigger_are_kept_and_warned_about(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    _mixed(workspace)

    with caplog.at_level(logging.WARNING, logger="train_zimage_lora"):
        trainer = _dataset(workspace, dataset={"trigger": "", "default_caption": ""})

    assert _captions(trainer) == {"empty": "", "named": "a photo of person"}
    assert "Detected 1 empty captions and no trigger token." in caplog.text


def test_no_warning_is_raised_when_a_trigger_names_the_concept(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    _mixed(workspace)

    with caplog.at_level(logging.WARNING, logger="train_zimage_lora"):
        _dataset(workspace, dataset={"default_caption": ""})

    assert "empty captions" not in caplog.text


def test_repeats_multiply_the_samples_and_batch_size_groups_them(workspace: Workspace) -> None:
    workspace.add_images(3)

    trainer = _dataset(workspace, dataset={"repeats": 2}, train={"batch_size": 4})

    assert len(trainer.train_dataset) == 6
    assert len(trainer.train_dataloader) == 2
    assert next(iter(trainer.train_dataloader))["pixel_values"].shape == (4, 3, 16, 16)


def test_center_crop_makes_square_samples_and_without_it_the_long_side_stays(workspace: Workspace) -> None:
    write_image(workspace.data / "wide.png", "x", size=(24, 20))

    cropped = _dataset(workspace)
    loose = _dataset(workspace, dataset={"center_crop": False})

    assert cropped.train_dataset[0]["pixel_values"].shape == (3, 16, 16)
    assert loose.train_dataset[0]["pixel_values"].shape == (3, 16, 19)


def test_random_flip_adds_the_flip_transform(workspace: Workspace) -> None:
    workspace.add_images(1)

    plain = _dataset(workspace)
    flipped = _dataset(workspace, dataset={"random_flip": True})

    kinds = [type(t) for t in flipped.train_dataset.transform.transforms]
    assert transforms.RandomHorizontalFlip in kinds
    assert transforms.RandomHorizontalFlip not in [type(t) for t in plain.train_dataset.transform.transforms]


def test_a_folder_without_images_is_refused(workspace: Workspace) -> None:
    workspace.data.mkdir()

    with pytest.raises(ValueError, match=r"No images found in "):
        _dataset(workspace)
