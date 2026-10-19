from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pytest
import torch

from maltrainer.data.qwen_dataset import QwenImage21Dataset, align_to_multiple

from .conftest import RED


@pytest.mark.parametrize(
    ("value", "multiple", "expected"), [(1, 16, 16), (16, 16, 16), (17, 16, 32), (512, 16, 512), (20, 8, 24)]
)
def test_a_side_is_rounded_up_to_the_multiple(value: int, multiple: int, expected: int) -> None:
    assert align_to_multiple(value, multiple) == expected


def test_images_are_found_in_nested_folders_sorted_by_path(
    data_root: Path, make_image: Callable[..., Path], write_file: Callable[[str, str], Path]
) -> None:
    for rel in ["b.png", "a.JPG", "people/c.webp", "people/ben/d.jpeg"]:
        make_image(rel)
    write_file("e.gif", "skipped")
    write_file("notes.md", "skipped")

    dataset = QwenImage21Dataset(data_root)

    assert [path.relative_to(data_root).as_posix() for path in dataset.image_paths] == [
        "a.JPG",
        "b.png",
        "people/ben/d.jpeg",
        "people/c.webp",
    ]


def test_a_missing_folder_is_an_empty_dataset_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="maltrainer.data.qwen_dataset"):
        dataset = QwenImage21Dataset(tmp_path / "missing")

    assert len(dataset) == 0
    assert "does not exist" in caplog.text


@pytest.mark.parametrize(
    ("caption", "trigger", "default", "expected"),
    [
        (" a cat \n", None, "", "a cat"),
        ("a cat", "sks", "", "sks a cat"),
        ("an SKS cat", "sks", "", "an SKS cat"),
        (None, "sks", "a photo", "sks a photo"),
        (None, "sks", "a photo of sks", "a photo of sks"),
        (None, "sks", "a photo of SKS", "a photo of SKS"),
        (None, "", "a photo", "a photo"),
        (None, "", "", ""),
        (None, " sks ", "", "sks"),
        (None, None, " a photo ", "a photo"),
        (None, None, "", ""),
        ("  ", "sks", "", "sks"),
    ],
)
def test_the_caption_falls_back_to_the_trigger_and_default_caption(
    data_root: Path,
    make_image: Callable[..., Path],
    caption: str | None,
    trigger: str | None,
    default: str,
    expected: str,
) -> None:
    make_image("a.png", caption=caption)

    dataset = QwenImage21Dataset(data_root, trigger_token=trigger, default_caption=default)

    assert dataset.captions == [expected]


def test_an_unreadable_caption_counts_as_missing(
    data_root: Path,
    make_image: Callable[..., Path],
    write_file: Callable[[str, bytes], Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    make_image("a.png")
    write_file("a.txt", b"\xff\xfe\xfa not utf-8")

    with caplog.at_level(logging.WARNING, logger="maltrainer.data.qwen_dataset"):
        dataset = QwenImage21Dataset(data_root, trigger_token="sks")

    assert dataset.captions == ["sks"]
    assert "Error reading caption" in caplog.text


@pytest.mark.parametrize(("repeats", "copies"), [(1, 1), (0, 1), (-2, 1), (3, 3)])
def test_each_image_is_repeated_in_place(
    data_root: Path, make_image: Callable[..., Path], repeats: int, copies: int
) -> None:
    make_image("a.png", caption="first")
    make_image("b.png", caption="second")

    dataset = QwenImage21Dataset(data_root, repeats=repeats)

    assert [path.name for path in dataset.image_paths] == ["a.png"] * copies + ["b.png"] * copies
    assert dataset.captions == ["first"] * copies + ["second"] * copies


@pytest.mark.parametrize(("resolution", "height", "width"), [(20, 32, 32), (512, 512, 512), ((20, 40), 32, 48)])
def test_the_resolution_is_aligned_to_sixteen_pixels(
    data_root: Path, resolution: int | tuple[int, int], height: int, width: int
) -> None:
    dataset = QwenImage21Dataset(data_root, resolution=resolution)

    assert (dataset.height, dataset.width) == (height, width)


def test_a_sample_is_the_image_stretched_to_the_resolution_in_zero_one(
    data_root: Path, make_image: Callable[..., Path]
) -> None:
    path = make_image("a.png", size=(40, 24), color=RED, caption="red")

    sample = QwenImage21Dataset(data_root, resolution=(16, 48))[0]

    pixels = sample["pixel_values"]
    assert pixels.shape == (3, 16, 48)
    assert pixels.dtype == torch.float32
    assert torch.equal(pixels[0], torch.ones(16, 48))
    assert torch.equal(pixels[1:], torch.zeros(2, 16, 48))
    assert sample["caption"] == "red"
    assert sample["image_path"] == str(path)


def test_an_undecodable_image_becomes_a_black_sample(
    data_root: Path, write_file: Callable[[str, bytes], Path], caplog: pytest.LogCaptureFixture
) -> None:
    write_file("broken.png", b"not a png")

    with caplog.at_level(logging.ERROR, logger="maltrainer.data.qwen_dataset"):
        sample = QwenImage21Dataset(data_root, resolution=(16, 32))[0]

    assert torch.equal(sample["pixel_values"], torch.zeros(3, 16, 32))
    assert "Failed to load image" in caplog.text


def test_images_in_hidden_folders_are_not_trained_on(data_root: Path, make_image: Callable[..., Path]) -> None:
    for rel in ["a.png", "people/b.png", ".ipynb_checkpoints/a-checkpoint.png", "people/.trash/c.png"]:
        make_image(rel)

    dataset = QwenImage21Dataset(data_root)

    assert [path.name for path in dataset.image_paths] == ["a.png", "b.png"]
