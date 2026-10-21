"""``mutation.main``: choosing targets, printing, the exit code (the pool and the runs are stubs here)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maltrainer_quality import mutation, mutation_pool
from maltrainer_quality.mutation import Report
from maltrainer_quality.mutation_ops import Site
from maltrainer_quality.mutation_pool import Worker
from maltrainer_quality.mutation_targets import TARGETS, Target


class FakePool:
    """Stands in for the worker pool: no processes, the pool is only a context manager here."""

    def __init__(self, _size: int) -> None:
        pass

    def __enter__(self) -> FakePool:
        return self

    def __exit__(self, *_error: object) -> None:
        return None


def fake_run_target(_pool: object, target: Target, _root: Path) -> Report:
    survivors = (Site(0, 1, "a  ->  b"),) if target.path == TARGETS[1].path else ()
    return Report(target, 10, survivors, 1.0)


def test_main_runs_the_chosen_targets_prints_each_and_fails_when_one_is_below_its_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mutation, "Pool", FakePool)
    monkeypatch.setattr(mutation, "run_target", fake_run_target)
    result = tmp_path / "result.json"

    assert mutation.main(["--all", "--json", str(result)]) == 1

    out = capsys.readouterr().out
    assert out.count(" killed, ") == len(TARGETS) and "BELOW FLOOR" in out
    assert set(json.loads(result.read_text(encoding="utf-8"))) == {target.path for target in TARGETS}
    assert mutation.main([TARGETS[0].path]) == 0


def test_main_with_changed_runs_what_changed_and_says_so_when_nothing_did(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mutation, "Pool", FakePool)
    monkeypatch.setattr(mutation, "run_target", fake_run_target)
    monkeypatch.setattr(
        mutation, "changed_files", lambda _root, base: {"README.md"} if base == "origin/main" else {TARGETS[0].path}
    )

    assert mutation.main(["--changed"]) == 0
    assert "no mutation target module or its tests changed" in capsys.readouterr().out
    assert mutation.main(["--changed", "--base", "dev"]) == 0
    assert capsys.readouterr().out.count(" killed, ") == 1


def test_a_worker_that_never_says_ready_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mutation_pool, "START_TIMEOUT", 0.0)

    with pytest.raises(RuntimeError, match="did not start"):
        Worker(())
