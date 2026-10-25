"""scripts/merge_lora_into_model.py with other checkpoints: weighted sum, add difference and the warnings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tests.scripts.merge_into_model_world import P, model_file, near, run


def test_a_weighted_sum_is_one_minus_t_of_the_base_plus_t_of_themodel_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = model_file(tmp_path / "a.safetensors", 1)
    b = near(tmp_path / "b.safetensors", a, 0.1, 2)
    out = tmp_path / "out.safetensors"

    assert run(monkeypatch, str(a), str(out), "--model", f"{b}:0.3") == 0

    ta, tb, merged = load_file(str(a)), load_file(str(b)), load_file(str(out))
    for key in (f"{P}blocks.0.lin.weight", f"{P}blocks.0.norm.scale"):
        torch.testing.assert_close(merged[key], 0.7 * ta[key] + 0.3 * tb[key])
    assert torch.equal(merged[f"{P}step"], ta[f"{P}step"])  # integers come from the base


def test_add_difference_adds_strength_times_the_finetunes_change_and_loras_go_on_top(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = model_file(tmp_path / "raw.safetensors", 1)
    finetune = near(tmp_path / "ft.safetensors", raw, 0.05, 2)
    turbo = near(tmp_path / "turbo.safetensors", raw, 0.1, 3)
    lora = tmp_path / "lora.safetensors"
    down, up = torch.ones(1, 6), torch.full((4, 1), 0.5)
    save_file(
        {"diffusion_model.blocks.0.lin.lora_A.weight": down, "diffusion_model.blocks.0.lin.lora_B.weight": up},
        str(lora),
    )
    out = tmp_path / "out.safetensors"

    assert run(monkeypatch, str(turbo), str(out), f"{lora}:2", "--add-diff", f"{finetune},{raw}:0.8") == 0
    printed = capsys.readouterr().out
    assert "  base turbo.safetensors\n  difference ft.safetensors - raw.safetensors at strength 0.8\n" in printed

    t, f, r, merged = (load_file(str(p)) for p in (turbo, finetune, raw, out))
    key = f"{P}blocks.0.lin.weight"
    torch.testing.assert_close(merged[key], t[key] + 0.8 * (f[key] - r[key]) + 2 * (up @ down))
    norm = f"{P}blocks.0.norm.scale"
    torch.testing.assert_close(merged[norm], t[norm] + 0.8 * (f[norm] - r[norm]))
    with safe_open(str(out), framework="pt") as handle:
        recipe = json.loads(handle.metadata()["maltrainer_merge_into_model"])
    assert [(d["model"]["file"], d["base"]["file"], d["strength"]) for d in recipe["add_difference"]] == [
        ("ft.safetensors", "raw.safetensors", 0.8)
    ]


def test_a_difference_whose_base_is_the_merge_base_itself_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = model_file(tmp_path / "raw.safetensors", 1)
    finetune = near(tmp_path / "ft.safetensors", raw, 0.05, 2)
    out = tmp_path / "out.safetensors"

    assert run(monkeypatch, str(raw), str(out), "--add-diff", f"{finetune},{raw}:1") == 0

    torch.testing.assert_close(
        load_file(str(out))[f"{P}blocks.0.lin.weight"], load_file(str(finetune))[f"{P}blocks.0.lin.weight"]
    )


def test_differences_between_the_models_are_warned_about_and_the_file_is_still_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = model_file(tmp_path / "a.safetensors", 1)
    other = (
        tmp_path / "other.safetensors"
    )  # another base model: unrelated values, a missing, a reshaped, an extra tensor
    save_file(
        {
            f"{P}blocks.0.lin.weight": torch.randn(4, 6) * 5,
            f"{P}blocks.0.norm.scale": torch.randn(7),
            f"{P}blocks.9.new.weight": torch.ones(2, 2),
        },
        str(other),
    )
    out = tmp_path / "out.safetensors"

    assert run(monkeypatch, str(a), str(out), "--model", f"{other}:1.5") == 0

    err = capsys.readouterr().err
    assert "the model weights sum to 1.5: outside [0, 1]" in err
    assert "other.safetensors holds in another shape 1 tensors (the base's value is kept there)" in err
    assert "other.safetensors adds 1 tensors (ignored), e.g. ['blocks.9.new.weight']" in err
    assert "away from the base: a different base model?" in err
    merged, base = load_file(str(out)), load_file(str(a))
    torch.testing.assert_close(merged[f"{P}blocks.0.norm.scale"], base[f"{P}blocks.0.norm.scale"])  # -0.5 a + 1.5 a


def test_a_difference_far_from_its_base_or_built_from_fp8_is_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = model_file(tmp_path / "a.safetensors", 1)
    unrelated = model_file(tmp_path / "unrelated.safetensors", 5, scale=3.0)
    scale = torch.full((4, 1), 0.01)
    fp8 = model_file(tmp_path / "fp8.safetensors", 1, **{f"{P}blocks.0.lin.weight_scale": scale})
    tensors = load_file(str(fp8))
    tensors[f"{P}blocks.0.lin.weight"] = (tensors[f"{P}blocks.0.lin.weight"] / scale).to(torch.float8_e4m3fn)
    save_file(tensors, str(fp8))

    assert run(monkeypatch, str(a), str(tmp_path / "out.safetensors"), "--add-diff", f"{unrelated},{fp8}") == 0

    err = capsys.readouterr().err
    assert "unrelated.safetensors is" in err and "away from fp8.safetensors: is it really a finetune of it?" in err
    assert "fp8.safetensors is fp8: its rounding noise goes into unrelated.safetensors - fp8.safetensors" in err


def test_it_shows_its_progress_through_the_tensors_and_the_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = model_file(tmp_path / "a.safetensors", 1)
    b = near(tmp_path / "b.safetensors", a, 0.1, 2)

    assert run(monkeypatch, str(a), str(tmp_path / "out.safetensors"), "--model", f"{b}:0.5") == 0

    captured = capsys.readouterr()
    assert "  base a.safetensors at weight 0.5\n  model b.safetensors at weight 0.5\n" in captured.out
    assert "Merging into a.safetensors" in captured.err and "3/3" in captured.err  # the bar, over all 3 tensors
    assert "Hashing the 2 input files for the record" in captured.out and "Writing out.safetensors..." in captured.out
    sizes = sum(p.stat().st_size for p in (a, b))
    assert "Hashing: 100%" in captured.err and f"{sizes}/{sizes}" in captured.err  # every byte of both files
