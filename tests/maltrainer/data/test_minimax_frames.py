from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
from PIL import Image

from maltrainer.data.minimax_dataset import (
    align_to_32,
    audio_latent_frames,
    bucket_size,
    frames_tensor,
    grid_frames,
    load_image,
    resize_and_crop,
)

from .conftest import BLUE, RED


@pytest.mark.parametrize(("value", "expected"), [(1, 32), (32, 32), (33, 64), (512, 512)])
def test_a_side_is_rounded_up_to_32_pixels(value: int, expected: int) -> None:
    assert align_to_32(value) == expected


@pytest.mark.parametrize(
    ("frames", "expected"), [(0, 0), (4, 0), (5, 5), (21, 5), (22, 22), (38, 22), (39, 39), (73, 73), (100, 90)]
)
def test_clip_lengths_snap_down_to_the_17k_plus_5_grid(frames: int, expected: int) -> None:
    assert grid_frames(frames) == expected


@pytest.mark.parametrize(("frames", "expected"), [(1, 2), (5, 8), (24, 40), (56, 93), (107, 178)])
def test_audio_latents_are_rounded_from_the_duration_at_40_per_second(frames: int, expected: int) -> None:
    assert audio_latent_frames(frames) == expected


@pytest.mark.parametrize(
    ("width", "height", "resolution", "step", "expected"),
    [
        (40, 24, 32, 32, (32, 32)),
        (40, 24, 64, 32, (64, 32)),
        (40, 24, 64, 16, (80, 48)),
        (1000, 500, 512, 32, (704, 352)),
        (10, 1000, 32, 32, (32, 320)),
        (512, 512, 512, 32, (512, 512)),
    ],
)
def test_buckets_keep_the_aspect_at_about_the_target_area_on_the_grid(
    width: int, height: int, resolution: int, step: int, expected: tuple[int, int]
) -> None:
    assert bucket_size(width, height, resolution, step) == expected


def test_resize_and_crop_covers_the_target_then_crops_the_centre() -> None:
    image = Image.new("RGB", (40, 24), RED)
    image.paste(BLUE, (20, 0, 40, 24))

    result = resize_and_crop(image, (16, 16))

    assert result.size == (16, 16)
    assert result.getpixel((0, 8)) == RED
    assert result.getpixel((15, 8)) == BLUE


def test_an_image_that_already_covers_the_target_is_only_cropped() -> None:
    rows = np.repeat(np.arange(24, dtype=np.uint8)[:, None] * 10, 40, axis=1)
    image = Image.fromarray(np.stack([rows] * 3, axis=-1))

    result = np.asarray(resize_and_crop(image, (40, 16)))

    assert result.shape == (16, 40, 3)
    assert list(result[:, 0, 0]) == [row * 10 for row in range(4, 20)]


def test_an_image_already_at_the_target_comes_back_unchanged() -> None:
    image = Image.new("RGB", (32, 32), RED)

    assert np.array_equal(np.asarray(resize_and_crop(image, (32, 32))), np.asarray(image))


@pytest.mark.parametrize(
    ("resolution", "mode", "shape"),
    [
        (32, "bucket", (3, 32, 32)),
        (64, "bucket", (3, 32, 64)),
        (20, "stretch", (3, 32, 32)),
        (64, "stretch", (3, 64, 64)),
    ],
)
def test_stills_load_as_aspect_buckets_or_stretched_squares(
    make_image: Callable[..., Path], resolution: int, mode: str, shape: tuple[int, int, int]
) -> None:
    path = make_image("a.png", size=(40, 24), color=(255, 0, 51))

    pixels = load_image(path, resolution, mode)

    assert pixels.shape == shape
    assert pixels.dtype == torch.float32
    assert torch.allclose(pixels[:, 0, 0], torch.tensor([1.0, 0.0, 0.2]))


def test_a_greyscale_still_loads_as_three_rgb_channels(data_root: Path) -> None:
    path = data_root / "grey.png"
    Image.new("L", (32, 32), 51).save(path)

    pixels = load_image(path, 32)

    assert pixels.shape == (3, 32, 32)
    assert torch.allclose(pixels, torch.full((3, 32, 32), 0.2))


def test_frames_become_a_batch_of_one_channels_first_video_in_zero_one() -> None:
    frames = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    frames[1, 2, 3] = [255, 0, 51]

    video = frames_tensor(frames)

    assert video.shape == (1, 3, 2, 3, 4)
    assert video.dtype == torch.float32
    assert torch.allclose(video[0, :, 1, 2, 3], torch.tensor([1.0, 0.0, 0.2]))
    assert video.sum().item() == pytest.approx(1.2)
