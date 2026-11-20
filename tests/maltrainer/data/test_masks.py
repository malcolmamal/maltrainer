"""Mask files: finding them, reading them, and putting them on a latent grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from maltrainer.data.masks import aspect_ok, is_mask_file, latent_mask, mask_for, mask_problem, read_mask


def _save(path: Path, array: np.ndarray, mode: str | None) -> Path:
    Image.fromarray(array, mode).save(path)
    return path


def test_a_mask_is_found_beside_its_picture_in_any_format_and_case(tmp_path: Path) -> None:
    (tmp_path / "beach.jpg").write_bytes(b"x")
    assert mask_for(tmp_path / "beach.jpg") is None

    _save(tmp_path / "Beach-MaskLabel.PNG", np.zeros((4, 4), np.uint8), "L")

    assert mask_for(tmp_path / "beach.jpg") == tmp_path / "Beach-MaskLabel.PNG"
    assert is_mask_file("x/beach-masklabel.png") and not is_mask_file("beach.png")


def test_masks_read_as_brightness_with_transparent_as_black(tmp_path: Path) -> None:
    gray = _save(tmp_path / "g.png", np.full((2, 2), 200, np.uint8), "L")
    rgba = np.zeros((2, 2, 4), np.uint8)
    rgba[0, 0] = (255, 255, 255, 255)
    rgba[0, 1] = (255, 255, 255, 0)  # white but transparent
    rgb = _save(tmp_path / "a.png", rgba, "RGBA")
    wide = _save(tmp_path / "w.png", np.array([[0, 60000]], np.uint16), None)
    small = _save(tmp_path / "s.png", np.array([[0, 100]], np.uint16), None)
    palette = Image.new("P", (2, 1))
    palette.info["transparency"] = 0
    palette.save(tmp_path / "p.png", transparency=0)

    assert np.asarray(read_mask(gray)).tolist() == [[200, 200], [200, 200]]
    assert np.asarray(read_mask(rgb))[0].tolist() == [255, 0]
    assert np.asarray(read_mask(wide))[0].tolist() == [0, 233]
    assert np.asarray(read_mask(small))[0].tolist() == [0, 100]
    assert np.asarray(read_mask(tmp_path / "p.png")).tolist() == [[0, 0]]


def test_the_shape_of_a_mask_must_match_its_picture() -> None:
    assert aspect_ok((10, 20), (50, 100)) and aspect_ok((100, 100), (101, 100))
    assert not aspect_ok((20, 10), (50, 100)) and not aspect_ok((0, 10), (10, 10))


def test_a_mask_that_cannot_be_used_says_why(tmp_path: Path) -> None:
    good = _save(tmp_path / "good.png", np.full((8, 4), 255, np.uint8), "L")
    black = _save(tmp_path / "black.png", np.zeros((8, 4), np.uint8), "L")
    (tmp_path / "bad.png").write_bytes(b"not an image")

    assert mask_problem(good, (4, 8)) == ""
    assert "isn't the shape" in mask_problem(good, (8, 4))
    assert "all black" in mask_problem(black, (4, 8))
    assert mask_problem(tmp_path / "bad.png", (4, 8)) == "can't be read"


def test_a_mask_lands_on_the_latent_grid_with_the_pictures_crop(tmp_path: Path) -> None:
    array = np.zeros((16, 32), np.uint8)
    array[:, :16] = 255  # the left half of a 32 x 16 picture is white
    path = _save(tmp_path / "m.png", array, "L")

    wide = latent_mask(path, (32, 16), (32, 16), (2, 4))
    cropped = latent_mask(path, (32, 16), (16, 16), (2, 2))  # the picture was centre-cropped to a square

    assert wide.shape == (1, 1, 2, 4) and wide[0, 0].tolist() == [[1.0, 1.0, 0.0, 0.0]] * 2
    assert cropped[0, 0].tolist() == [[1.0, 0.0]] * 2  # the crop kept columns 8..24: half white, half black


def test_a_mask_saved_smaller_than_its_picture_is_stretched(tmp_path: Path) -> None:
    path = _save(tmp_path / "m.png", np.full((2, 4), 255, np.uint8), "L")

    mask = latent_mask(path, (16, 8), (16, 8), (1, 2))

    assert torch.equal(mask, torch.ones(1, 1, 1, 2))
