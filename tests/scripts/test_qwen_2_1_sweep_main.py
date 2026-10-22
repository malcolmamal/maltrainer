"""scripts/qwen_2_1_sweep_checkpoints.py main(): one image per checkpoint and seed, plus a contact sheet (CPU fakes)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import qwen_2_1_sweep_checkpoints as sweep
from tests.scripts.qwen_fakes import install, write_checkpoint, write_config


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *flags: str, checkpoints: bool = True) -> Path:
    install(monkeypatch, sweep)
    out = tmp_path / "run"
    out.mkdir()
    if checkpoints:
        for i, name in enumerate(("lora_step_10", "lora", "lora_step_5")):
            write_checkpoint(out / f"{name}.safetensors", i + 1)
    config = write_config(tmp_path / "config.yaml", out)
    sheets = tmp_path / "sheets"
    argv = [
        "sweep",
        "--config",
        str(config),
        "--prompt",
        "a cat",
        "--outdir",
        str(sheets),
        "--steps",
        "2",
        "--size",
        "64",
    ]
    monkeypatch.setattr("sys.argv", [*argv, *flags])
    sweep.main()
    return sheets


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image)


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def test_every_checkpoint_and_seed_gets_an_image_in_step_order_with_the_base_model_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheets = _run(monkeypatch, tmp_path, "--seeds", "1", "2")

    names = sorted(p.name for p in sheets.glob("*_seed*.png"))
    assert names == sorted(f"{label}_seed{s}.png" for label in ("base", "step_5", "step_10", "final") for s in (1, 2))
    out = capsys.readouterr().out
    assert "3 checkpoints: step 5, step 10, final" in out and "adapter attached: 1 modules" in out
    assert (
        out.index("base (no LoRA) done")
        < out.index("step 5 done")
        < out.index("step 10 done")
        < out.index("final done")
    )
    assert f"wrote 8 images + {sheets / 'contact_sheet.png'}" in out


def test_the_contact_sheet_has_a_row_per_checkpoint_and_a_column_per_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sheets = _run(monkeypatch, tmp_path, "--seeds", "1", "2")

    assert _size(sheets / "contact_sheet.png") == (2 * (64 + 4) + 4, 4 * (64 + 4 + 22) + 4)


def test_without_the_base_row_only_the_checkpoints_are_rendered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sheets = _run(monkeypatch, tmp_path, "--no-base")

    assert sorted(p.name for p in sheets.glob("*_seed*.png")) == [
        "final_seed1.png",
        "step_10_seed1.png",
        "step_5_seed1.png",
    ]
    assert _size(sheets / "contact_sheet.png") == (68 + 4, 3 * 90 + 4)


def test_a_checkpoint_changes_the_image_and_a_lora_scale_of_zero_gives_the_base_image_for_every_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sheets = _run(monkeypatch, tmp_path)
    tuned = _pixels(sheets / "step_5_seed1.png")

    assert not np.array_equal(_pixels(sheets / "base_seed1.png"), tuned)

    other = tmp_path / "again"
    other.mkdir()
    zero = _run(monkeypatch, other, "--lora-scale", "0")
    assert np.array_equal(_pixels(zero / "step_5_seed1.png"), _pixels(zero / "base_seed1.png"))
    assert np.array_equal(_pixels(zero / "final_seed1.png"), _pixels(zero / "base_seed1.png"))


def test_a_run_folder_without_checkpoints_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=r"no checkpoints in .*run"):
        _run(monkeypatch, tmp_path, checkpoints=False)
