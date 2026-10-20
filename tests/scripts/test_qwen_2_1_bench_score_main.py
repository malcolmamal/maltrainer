"""scripts/qwen_2_1_bench_score.py main(): likeness, adherence, the judge, the dataset and refusals.

CLIP-H, ArcFace and the Qwen3-VL judge are replaced by colour-based stand-ins (the real ones are model files
that live outside the repository); everything the script computes from their outputs is the real code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tests.scripts.score_world import ORANGE, RED, YELLOW, like, make_bench, run_score, summary_row


def test_likeness_is_scaled_between_the_base_model_and_a_real_photo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records, summary = run_score(monkeypatch, make_bench(tmp_path))

    floor = like(YELLOW)
    assert summary["face_backend"] == "fake"
    assert summary["reference"]["likeness_floor_base"] == pytest.approx(floor)
    assert summary["reference"]["likeness_ceiling_real"] == pytest.approx(1.0)
    assert summary["reference"]["training_faces"] == 3
    assert summary["reference"]["training_pairwise_clip"] == pytest.approx(1.0)
    for run, step, colour in (("base", "base", YELLOW), ("adamw", "5", ORANGE), ("adamw", "final", RED)):
        row = summary_row(summary, run, step)
        assert row["likeness"] == pytest.approx(like(colour))
        assert row["likeness_norm"] == pytest.approx((like(colour) - floor) / (1 - floor), abs=1e-6)
        assert (row["n"], row["face_rate"]) == (3, 1.0)
    assert {r["face_found"] for r in records} == {True}


def test_adherence_is_the_judges_mean_over_the_checks_minus_the_base_models_for_the_same_prompt_and_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    records, summary = run_score(monkeypatch, make_bench(tmp_path))

    def judged(colour: tuple[int, ...]) -> float:
        return (colour[0] / 255 + colour[1] / 255 + 0.1) / 2

    first = next(r for r in records if (r["run"], r["step"], r["prompt"], r["seed"]) == ("adamw", "5", "p1", 1))
    assert first["checks"] == {"Is it red?": pytest.approx(200 / 255), "Is it bright?": pytest.approx(50 / 255 + 0.1)}
    assert first["judge"] == pytest.approx(judged(ORANGE))
    assert first["adherence"] == pytest.approx(judged(ORANGE) - judged(YELLOW))
    assert summary_row(summary, "adamw", "final")["adherence"] == pytest.approx(judged(RED) - judged(YELLOW))
    assert summary_row(summary, "base", "base")["adherence"] == pytest.approx(0.0)
    assert summary_row(summary, "adamw", "5")["per_prompt"]["p2"] == {
        "likeness": pytest.approx(like(ORANGE)),
        "judge": None,
        "adherence": None,
    }
    unchecked = next(r for r in records if r["prompt"] == "p2")
    assert unchecked["checks"] == {} and unchecked["judge"] is None and unchecked["adherence"] is None


def test_without_the_judge_nothing_is_asked_and_the_judged_columns_stay_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records, summary = run_score(monkeypatch, make_bench(tmp_path), "--no-judge")

    assert all("judge" not in r and "adherence" not in r for r in records)
    assert {(r["judge"], r["adherence"]) for r in summary["rows"]} == {(None, None)}
    assert "   -  " in capsys.readouterr().out


def test_a_dataset_given_on_the_command_line_replaces_the_one_in_the_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bench = make_bench(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    for i in range(4):
        Image.new("RGB", (16, 16), ORANGE).save(other / f"o{i}.png")

    _, summary = run_score(monkeypatch, bench, "--no-judge", "--dataset", str(other))

    assert summary["reference"]["training_faces"] == 4


def test_a_training_set_with_fewer_than_three_faces_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bench = make_bench(tmp_path, faces=2)

    with pytest.raises(SystemExit, match="only 2 faces found in the training set"):
        run_score(monkeypatch, bench)

    assert not (bench / "summary.json").exists()


def test_judging_reports_its_progress_every_25_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bench = make_bench(tmp_path, {("base", "base"): YELLOW, **{(f"r{i}", "final"): RED for i in range(8)}})

    run_score(monkeypatch, bench)

    assert "  judged 25/27" in capsys.readouterr().out
