"""scripts/qwen_2_1_bench_score.py: helpers and the CLIP-H wrapper's image handling, without the model.

``ClipVisionH.__init__`` loads a 632M-parameter vision tower from a ComfyUI folder; it is not run here.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import qwen_2_1_bench_score as score


def _blank(path: Path, size: tuple[int, int] = (8, 8)) -> Path:
    Image.new("RGB", size, "gray").save(path)
    return path


def test_l2_scales_vectors_to_unit_length_and_leaves_zero_alone() -> None:
    out = score.l2(np.array([[3.0, 4.0], [0.0, 0.0]]))

    assert out[0].tolist() == pytest.approx([0.6, 0.8]) and out[1].tolist() == [0.0, 0.0]


def test_steps_order_base_first_numbers_by_value_and_final_last() -> None:
    assert sorted(["final", "10", "base", "5"], key=score.step_order) == ["base", "5", "10", "final"]


def test_training_images_are_every_image_below_the_folder_in_name_order(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    for name in ("b.jpg", "a.png", "sub/c.WEBP"):
        _blank(tmp_path / name)
    (tmp_path / "notes.txt").touch()
    (tmp_path / "d.gif").touch()

    images = score.load_training_images(tmp_path)

    assert len(images) == 3 and {i.mode for i in images} == {"RGB"}


def test_an_image_is_scaled_to_cover_224_centre_cropped_and_normalised_by_the_clip_statistics() -> None:
    clip = object.__new__(score.ClipVisionH)

    wide = clip._prep(Image.new("RGB", (448, 224), (255, 0, 0)))
    small = clip._prep(Image.new("RGB", (100, 50), (0, 0, 0)))

    assert wide.shape == small.shape == (3, 224, 224) and wide.dtype == np.float32
    assert wide[:, 0, 0].tolist() == pytest.approx(
        ((np.array([1.0, 0, 0]) - score.CLIP_MEAN) / score.CLIP_STD).tolist()
    )
    assert small[:, 100, 100].tolist() == pytest.approx((-score.CLIP_MEAN / score.CLIP_STD).tolist())


def test_images_go_through_the_model_in_batches_of_32_and_come_back_unit_length() -> None:
    clip = object.__new__(score.ClipVisionH)
    clip.device = "cpu"
    batches: list[tuple[int, torch.dtype]] = []

    def model(pixel_values: torch.Tensor) -> SimpleNamespace:
        batches.append((pixel_values.shape[0], pixel_values.dtype))
        return SimpleNamespace(
            image_embeds=torch.arange(1, 5, dtype=torch.float16).repeat(pixel_values.shape[0], 1) * 5
        )

    clip.model = model
    out = clip([Image.new("RGB", (30, 30)) for _ in range(40)])

    assert batches == [(32, torch.float16), (8, torch.float16)]
    assert out.shape == (40, 4) and out.dtype == np.float32
    assert np.linalg.norm(out, axis=1) == pytest.approx(np.ones(40), abs=1e-3)
