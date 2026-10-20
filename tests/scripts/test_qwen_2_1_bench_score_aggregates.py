"""scripts/qwen_2_1_bench_score.py main(): diversity, memorization, missing faces and the shape of the summary."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tests.scripts.score_world import BLACK, COLOURS, ORANGE, RED, make_bench, run_score, summary_row, unit


def test_diversity_is_one_minus_the_mean_clip_similarity_between_the_seeds_of_a_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bench = make_bench(tmp_path)
    Image.new("RGB", (16, 16), (0, 0, 255)).save(bench / "adamw" / "5" / "p1_s2.png")

    _, summary = run_score(monkeypatch, bench, "--no-judge")

    similar = float(unit([*ORANGE, 255]) @ unit([0, 0, 255, 255]))
    assert summary_row(summary, "adamw", "5")["diversity"] == pytest.approx(1 - similar, abs=1e-6)
    assert summary_row(summary, "adamw", "final")["diversity"] == pytest.approx(0.0, abs=1e-6)


def test_memorization_is_the_best_clip_match_to_a_training_image_averaged_over_the_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, summary = run_score(monkeypatch, make_bench(tmp_path), "--no-judge")

    expected = float(unit([*ORANGE, 255]) @ unit([*RED, 255]))
    assert summary_row(summary, "adamw", "5")["memorization"] == pytest.approx(expected, abs=1e-6)
    assert summary_row(summary, "adamw", "final")["memorization"] == pytest.approx(1.0, abs=1e-6)


def test_an_image_without_a_face_counts_against_the_face_rate_and_not_the_likeness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bench = make_bench(tmp_path)
    Image.new("RGB", (16, 16), BLACK).save(bench / "adamw" / "final" / "p1_s2.png")

    records, summary = run_score(monkeypatch, bench, "--no-judge")

    row = summary_row(summary, "adamw", "final")
    lost = next(r for r in records if (r["run"], r["step"], r["prompt"], r["seed"]) == ("adamw", "final", "p1", 2))
    assert (lost["face_found"], lost["likeness"]) == (False, None)
    assert row["face_rate"] == pytest.approx(2 / 3) and row["likeness"] == pytest.approx(1.0)
    assert row["likeness_sem"] == pytest.approx(0.0, abs=1e-6)


def test_rows_list_the_base_first_then_each_run_by_step_with_final_last_and_the_table_is_printed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = make_bench(tmp_path, {**COLOURS, ("adamw", "10"): ORANGE, ("aaa", "final"): RED})

    _, summary = run_score(monkeypatch, bench, "--no-judge")

    assert [(r["run"], r["step"]) for r in summary["rows"]] == [
        ("base", "base"),
        ("aaa", "final"),
        ("adamw", "5"),
        ("adamw", "10"),
        ("adamw", "final"),
    ]
    out = capsys.readouterr().out
    assert "face backend: fake" in out and "training faces 3/3, likeness ceiling 1.0000" in out
    assert "        run  step  face%  likeness   norm  judge  adhere  divers   memo" in out
