"""scripts/qwen_2_1_bench_generate.py main(): the image set and manifest an optimizer benchmark is scored from."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import qwen_2_1_bench_generate as generate
from tests.scripts.qwen_fakes import install, write_checkpoint, write_config

PROMPTS = [{"id": "a", "prompt": "a cat", "checks": ["Is it a cat?"]}, {"id": "b", "prompt": "a dog", "checks": []}]


def _run_folder(root: Path, name: str, steps: tuple[str, ...]) -> Path:
    out = root / name
    out.mkdir()
    for i, step in enumerate(steps):
        write_checkpoint(out / (f"lora_step_{step}.safetensors" if step != "final" else "lora.safetensors"), i + 1)
    return out


def _generate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configs: list[Path], *flags: str, default_prompts: bool = False
) -> Path:
    install(monkeypatch, generate)
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(PROMPTS))
    outdir = tmp_path / "bench"
    argv = ["bench", "--configs", *map(str, configs), "--outdir", str(outdir), "--steps", "2", "--size", "64"]
    argv += [] if default_prompts else ["--prompts-json", str(prompts)]
    monkeypatch.setattr("sys.argv", [*argv, *flags])
    generate.main()
    return outdir


def _two_runs(tmp_path: Path) -> list[Path]:
    adamw = write_config(tmp_path / "adamw.yaml", _run_folder(tmp_path, "bench_adamw", ("5", "final")))
    prodigy = write_config(tmp_path / "prodigy.yaml", _run_folder(tmp_path, "bench_prodigy", ("final",)))
    return [adamw, prodigy]


def test_the_layout_is_base_plus_run_and_step_folders_one_png_per_prompt_and_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outdir = _generate(monkeypatch, tmp_path, _two_runs(tmp_path), "--seeds", "1", "2")

    folders = {p.parent.relative_to(outdir).as_posix() for p in outdir.rglob("*.png")}
    assert folders == {"base", "adamw/5", "adamw/final", "prodigy/final"}
    assert sorted(p.name for p in (outdir / "base").iterdir()) == ["a_s1.png", "a_s2.png", "b_s1.png", "b_s2.png"]
    out = capsys.readouterr().out
    assert "adamw: 5, final" in out and "prodigy: final" in out and "16 samples at 64px, 2 steps" in out
    assert f"wrote 16 images and {outdir / 'manifest.json'}" in out


def test_the_manifest_lists_prompts_seeds_runs_dataset_and_every_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outdir = _generate(monkeypatch, tmp_path, _two_runs(tmp_path), "--seeds", "3")

    manifest = json.loads((outdir / "manifest.json").read_text())
    assert (
        manifest["prompts"] == PROMPTS and manifest["seeds"] == [3] and (manifest["steps"], manifest["size"]) == (2, 64)
    )
    assert manifest["runs"] == {"adamw": ["5", "final"], "prodigy": ["final"]} and manifest["dataset"] == "none"
    records: list[dict[str, Any]] = manifest["images"]
    assert len(records) == 8
    assert {(r["run"], r["step"]) for r in records} == {
        ("base", "base"),
        ("adamw", "5"),
        ("adamw", "final"),
        ("prodigy", "final"),
    }
    assert {Path(r["path"]).as_posix() for r in records if r["run"] == "base"} == {"base/a_s3.png", "base/b_s3.png"}
    assert {Path(r["path"]).as_posix() for r in records if (r["run"], r["step"]) == ("adamw", "5")} == {
        "adamw/5/a_s3.png",
        "adamw/5/b_s3.png",
    }
    assert all((outdir / r["path"]).is_file() for r in records)


def test_the_built_in_prompt_set_is_used_without_a_prompts_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = write_config(tmp_path / "c.yaml", _run_folder(tmp_path, "bench_adamw", ("final",)))

    outdir = _generate(monkeypatch, tmp_path, [config], "--seeds", "1", default_prompts=True)

    manifest = json.loads((outdir / "manifest.json").read_text())
    assert manifest["prompts"] == generate.DEFAULT_PROMPTS and len(manifest["images"]) == 6


def test_runs_that_disagree_on_the_model_files_are_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first = write_config(tmp_path / "a.yaml", _run_folder(tmp_path, "bench_a", ("final",)))
    second = write_config(
        tmp_path / "b.yaml", _run_folder(tmp_path, "bench_b", ("final",)), vae_path="other.safetensors"
    )

    with pytest.raises(SystemExit, match="all runs must share model paths"):
        _generate(monkeypatch, tmp_path, [first, second])


def test_a_run_without_checkpoints_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = write_config(tmp_path / "a.yaml", _run_folder(tmp_path, "bench_empty", ()))

    with pytest.raises(SystemExit, match=r"no checkpoints in .*bench_empty"):
        _generate(monkeypatch, tmp_path, [config])
