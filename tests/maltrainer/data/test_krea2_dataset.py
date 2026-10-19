from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
import torch

from maltrainer.data.krea2_dataset import KREA2_GRID, load_krea2_image

from .conftest import BLUE, RED


def test_the_grid_is_sixteen_pixels() -> None:
    assert KREA2_GRID == 16


@pytest.mark.parametrize(
    ("size", "resolution", "mode", "shape"),
    [
        ((40, 24), 32, "bucket", (3, 16, 32)),
        ((40, 24), 64, "bucket", (3, 48, 80)),
        ((24, 40), 64, "bucket", (3, 80, 48)),
        ((40, 24), 40, "square", (3, 32, 32)),
        ((40, 24), 8, "square", (3, 16, 16)),
    ],
)
def test_buckets_keep_the_aspect_and_square_crops_on_the_sixteen_pixel_grid(
    make_image: Callable[..., Path], size: tuple[int, int], resolution: int, mode: str, shape: tuple[int, int, int]
) -> None:
    path = make_image("a.png", size=size)

    pixels = load_krea2_image(path, resolution, mode)

    assert pixels.shape == shape


def test_pixels_are_rgb_floats_in_zero_one(make_image: Callable[..., Path]) -> None:
    path = make_image("a.png", color=(255, 0, 51))

    pixels = load_krea2_image(path, 32)

    assert pixels.dtype == torch.float32
    assert torch.equal(pixels[0], torch.ones(16, 32))
    assert torch.equal(pixels[1], torch.zeros(16, 32))
    assert torch.allclose(pixels[2], torch.full((16, 32), 0.2))


def test_a_square_crop_is_taken_from_the_centre(make_image: Callable[..., Path]) -> None:
    path = make_image("a.png", size=(64, 32), color=RED, right=BLUE)

    pixels = load_krea2_image(path, 32, "square")

    assert torch.equal(pixels[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(pixels[:, 0, -1], torch.tensor([0.0, 0.0, 1.0]))
