from __future__ import annotations

import json
from pathlib import Path

import pytest

from maltrainer_quality import ratchet


def test_an_offender_missing_from_the_baseline_is_new() -> None:
    verdict = ratchet.judge({"a.py": 250}, {})

    assert verdict.new == {"a.py": 250}
    assert verdict.failed


def test_an_offender_at_its_baseline_passes() -> None:
    verdict = ratchet.judge({"a.py": 250}, {"a.py": 250})

    assert verdict == ratchet.Verdict()
    assert not verdict.failed


def test_an_offender_past_its_baseline_grew() -> None:
    assert ratchet.judge({"a.py": 251}, {"a.py": 250}).grew == {"a.py": (250, 251)}


def test_an_offender_below_its_baseline_is_loose() -> None:
    assert ratchet.judge({"a.py": 240}, {"a.py": 250}).loose == {"a.py": (250, 240)}


def test_an_entry_with_no_offender_is_loose() -> None:
    verdict = ratchet.judge({}, {"a.py": 250})

    assert verdict.loose == {"a.py": (250, None)}
    assert verdict.failed


def test_tightening_lowers_and_drops_but_never_adds_or_raises() -> None:
    over = {"shrunk.py": 240, "grown.py": 260, "new.py": 300}
    baseline = {"shrunk.py": 250, "grown.py": 250, "fixed.py": 250}

    assert ratchet.tightened(over, baseline) == {"grown.py": 250, "shrunk.py": 240}


def test_a_missing_baseline_file_is_empty(tmp_path: Path) -> None:
    assert ratchet.load(tmp_path / "absent.json") == {}


def test_a_saved_baseline_is_sorted_and_loads_back(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "baseline.json"

    ratchet.save(path, {"b.py": 2, "a.py": 1})

    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["a.py", "b.py"]
    assert ratchet.load(path) == {"a.py": 1, "b.py": 2}


def test_the_report_names_every_problem(capsys: pytest.CaptureFixture[str]) -> None:
    verdict = ratchet.Verdict(new={"n.py": 300}, grew={"g.py": (250, 260)}, loose={"l.py": (250, None)})

    ratchet.report(verdict, unit="lines", limit=200, command="gate")

    assert capsys.readouterr().out.splitlines() == [
        "n.py: 300 lines (limit 200)",
        "g.py: 260 lines, grew past its baseline of 250 (limit 200)",
        "l.py: baseline 250 is loose, within the limit",
        "Lower the baseline with `gate --tighten`.",
    ]


def test_a_shrunk_offender_is_reported_with_its_new_value(capsys: pytest.CaptureFixture[str]) -> None:
    ratchet.report(ratchet.Verdict(loose={"l.py": (250, 240)}), unit="lines", limit=200, command="gate")

    assert capsys.readouterr().out.splitlines()[0] == "l.py: baseline 250 is loose, now 240 lines"


def test_enforce_fails_on_a_loose_entry_until_tightened(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    ratchet.save(path, {"a.py": 250, "fixed.py": 250})
    options = {"unit": "lines", "limit": 200, "command": "gate"}

    assert ratchet.enforce({"a.py": 240}, path, tighten=False, **options) == 1
    assert ratchet.enforce({"a.py": 240}, path, tighten=True, **options) == 0
    assert ratchet.load(path) == {"a.py": 240}


def test_tightening_does_not_excuse_a_new_offender(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"

    assert ratchet.enforce({"new.py": 300}, path, tighten=True, unit="lines", limit=200, command="gate") == 1
    assert ratchet.load(path) == {}


def test_enforce_reports_what_judge_found(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "baseline.json"
    ratchet.save(path, {"grown.py": 250, "shrunk.py": 250})

    ratchet.enforce({"grown.py": 260, "shrunk.py": 240}, path, tighten=False, unit="lines", limit=200, command="g")

    assert capsys.readouterr().out.splitlines()[:2] == [
        "grown.py: 260 lines, grew past its baseline of 250 (limit 200)",
        "shrunk.py: baseline 250 is loose, now 240 lines",
    ]
