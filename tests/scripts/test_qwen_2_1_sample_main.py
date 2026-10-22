"""scripts/qwen_2_1_sample.py main(): one sampled image, with or without a LoRA (CPU fakes for the models)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import qwen_2_1_sample
from tests.scripts.qwen_fakes import install, write_checkpoint, write_config


def _sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, *flags: str) -> np.ndarray:
    install(monkeypatch, qwen_2_1_sample)
    config = write_config(tmp_path / "config.yaml", tmp_path / "out")
    out = tmp_path / "images" / f"{name}.png"
    argv = ["sample", "--config", str(config), "--prompt", "a cat", "--out", str(out), "--steps", "3", *flags]
    monkeypatch.setattr("sys.argv", argv)
    qwen_2_1_sample.main()
    return np.asarray(Image.open(out))


def test_the_image_is_written_at_the_requested_size_into_a_created_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pixels = _sample(monkeypatch, tmp_path, "a", "--size", "64")

    assert pixels.shape == (64, 64, 3) and pixels.dtype == np.uint8
    out = capsys.readouterr().out
    assert "sampling 64x64 (16 image tokens), 3 steps" in out and f"wrote {tmp_path / 'images' / 'a.png'}" in out


@pytest.mark.parametrize(("size", "edge"), [(100, 96), (63, 32), (64, 64)])
def test_the_size_is_rounded_down_to_the_32_pixel_grid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, size: int, edge: int
) -> None:
    assert _sample(monkeypatch, tmp_path, "a", "--size", str(size)).shape[:2] == (edge, edge)


def test_the_same_seed_gives_the_same_image_and_another_seed_a_different_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = _sample(monkeypatch, tmp_path, "a", "--size", "64", "--seed", "1")
    again = _sample(monkeypatch, tmp_path, "b", "--size", "64", "--seed", "1")
    other = _sample(monkeypatch, tmp_path, "c", "--size", "64", "--seed", "2")

    assert np.array_equal(first, again) and not np.array_equal(first, other)


def test_a_lora_changes_the_image_and_a_scale_of_zero_gives_the_base_image_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lora = write_checkpoint(tmp_path / "lora.safetensors", 1)
    base = _sample(monkeypatch, tmp_path, "base", "--size", "64")
    capsys.readouterr()

    tuned = _sample(monkeypatch, tmp_path, "tuned", "--size", "64", "--lora", str(lora), "--lora-scale", "2")
    said = capsys.readouterr().out
    off = _sample(monkeypatch, tmp_path, "off", "--size", "64", "--lora", str(lora), "--lora-scale", "0")

    assert "applied LoRA lora.safetensors: 1 modules at scale 2.0" in said
    assert not np.array_equal(base, tuned) and np.array_equal(base, off)
