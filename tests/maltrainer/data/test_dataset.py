from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
import torch

from maltrainer.data.dataset import ZImageDataset

from .conftest import BLUE, RED


def _names(dataset: ZImageDataset) -> list[str]:
    return [path.name for path in dataset.image_paths]


def test_images_are_found_in_nested_folders_by_extension(
    data_root: Path, make_image: Callable[..., Path], write_file: Callable[[str, str], Path]
) -> None:
    for rel in ["a.png", "b.JPG", "people/c.jpeg", "people/ben/d.webp"]:
        make_image(rel)
    write_file("notes.txt", "not an image")
    write_file("e.gif", "skipped")

    dataset = ZImageDataset(str(data_root))

    assert sorted(_names(dataset)) == ["a.png", "b.JPG", "c.jpeg", "d.webp"]
    assert len(dataset) == 4


def test_a_folder_without_images_is_an_error(data_root: Path, write_file: Callable[[str, str], Path]) -> None:
    write_file("caption.txt", "lonely")

    with pytest.raises(ValueError, match="No images found"):
        ZImageDataset(str(data_root))


@pytest.mark.parametrize(
    ("caption", "trigger", "expected"),
    [
        ("  a red square \n", None, "a red square"),
        (None, None, ""),
        ("a red square", "sks", "sks a red square"),
        (None, "sks", "sks"),
        ("", "sks", "sks"),
    ],
)
def test_the_caption_is_the_sidecar_text_behind_the_trigger(
    data_root: Path, make_image: Callable[..., Path], caption: str | None, trigger: str | None, expected: str
) -> None:
    make_image("a.png", caption=caption)

    dataset = ZImageDataset(str(data_root), trigger_token=trigger)

    assert dataset.captions == [expected]
    assert dataset[0]["caption"] == expected


@pytest.mark.parametrize(("repeats", "length"), [(1, 2), (0, 2), (3, 6)])
def test_repeats_multiply_the_samples_with_their_captions(
    data_root: Path, make_image: Callable[..., Path], repeats: int, length: int
) -> None:
    make_image("a.png", caption="first")
    make_image("b.png", caption="second")

    dataset = ZImageDataset(str(data_root), repeats=repeats)

    assert len(dataset) == length
    assert dict(zip(_names(dataset), dataset.captions, strict=False)) == {"a.png": "first", "b.png": "second"}


def test_a_sample_is_a_centre_cropped_square_normalised_to_minus_one_one(
    data_root: Path, make_image: Callable[..., Path]
) -> None:
    make_image("a.png", color=RED)

    pixels = ZImageDataset(str(data_root), resolution=16)[0]["pixel_values"]

    assert pixels.shape == (3, 16, 16)
    assert pixels.dtype == torch.float32
    assert torch.equal(pixels[0], torch.ones(16, 16))
    assert torch.equal(pixels[1:], -torch.ones(2, 16, 16))


def test_without_centre_crop_the_short_side_is_resized_and_the_aspect_kept(
    data_root: Path, make_image: Callable[..., Path]
) -> None:
    make_image("a.png", size=(40, 24))

    pixels = ZImageDataset(str(data_root), resolution=16, center_crop=False)[0]["pixel_values"]

    assert pixels.shape == (3, 16, 26)


def test_random_flip_mirrors_some_samples_and_not_others(data_root: Path, make_image: Callable[..., Path]) -> None:
    make_image("a.png", size=(32, 32), color=RED, right=BLUE)
    dataset = ZImageDataset(str(data_root), resolution=32, random_flip=True)
    torch.manual_seed(0)

    left_reds = {bool(dataset[0]["pixel_values"][0, 0, 0] > 0) for _ in range(20)}

    assert left_reds == {True, False}


def test_without_random_flip_the_image_keeps_its_orientation(data_root: Path, make_image: Callable[..., Path]) -> None:
    make_image("a.png", size=(32, 32), color=RED, right=BLUE)
    dataset = ZImageDataset(str(data_root), resolution=32)
    torch.manual_seed(0)

    pixels = [dataset[0]["pixel_values"] for _ in range(5)]

    assert all(sample[0, 0, 0] == 1.0 and sample[2, 0, -1] == 1.0 for sample in pixels)


def test_images_in_hidden_folders_are_not_trained_on(data_root: Path, make_image: Callable[..., Path]) -> None:
    for rel in ["a.png", "people/b.png", ".ipynb_checkpoints/a-checkpoint.png", "people/.trash/c.png"]:
        make_image(rel)

    assert sorted(_names(ZImageDataset(str(data_root)))) == ["a.png", "b.png"]
