"""The worker's own pieces, in this process: the finder, the module purge, one job and the serve loop."""

from __future__ import annotations

import importlib
import sys
import threading
from multiprocessing import Pipe
from pathlib import Path
from typing import Any

import pytest

from maltrainer_quality import mutation_worker
from maltrainer_quality.mutation_ops import mutate
from maltrainer_quality.mutation_worker import READY, _MutantFinder, forget_project, run_job, serve

SOURCE = "VALUE = 2 + 3\n"


def a_module(root: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    (root / f"{name}.py").write_text(SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delitem(sys.modules, name, raising=False)


def test_the_finder_serves_the_mutated_code_for_its_module_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a_module(tmp_path, monkeypatch, "mutprobe_one")
    a_module(tmp_path, monkeypatch, "mutprobe_two")
    finder = _MutantFinder("mutprobe_one", mutate(SOURCE, 2, "mutprobe_one"))  # 2 + 3 -> 2 - 3
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    assert importlib.import_module("mutprobe_one").VALUE == -1
    assert importlib.import_module("mutprobe_two").VALUE == 5
    assert finder.find_spec("mutprobe_absent", None) is None
    assert _MutantFinder("mutprobe_absent", mutate(SOURCE, 0)).find_spec("mutprobe_absent", None) is None


def test_the_project_modules_are_forgotten_and_the_rest_are_kept() -> None:
    loaded: dict[str, object] = {
        name: object()
        for name in (
            "maltrainer.a",
            "maltrainer_quality.b",
            "backend.c",
            "tests.d",
            "train_e",
            "sample_pkg.f",
            "sample_pkg",
            "torch",
            "sample_other.g",
        )
    }

    forget_project("sample_pkg.module", loaded)

    assert set(loaded) == {"torch", "sample_other.g"}


def test_a_job_installs_the_mutant_for_the_run_and_removes_it_after(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_main(arguments: list[str]) -> int:
        seen["arguments"] = arguments
        seen["finders"] = [f for f in sys.meta_path if isinstance(f, _MutantFinder)]
        sys.stdout.write("pytest output\n")
        return 1

    monkeypatch.setattr(mutation_worker.pytest, "main", fake_main)
    monkeypatch.setattr(mutation_worker, "forget_project", lambda module: seen.setdefault("forgot", module))
    before = list(sys.meta_path)

    code, seconds, output = run_job("pkg.mod", SOURCE, 0, ("tests/a.py", "tests/b.py"))

    assert (code, output) == (1, "pytest output\n")
    assert seconds >= 0
    assert seen["arguments"] == ["tests/a.py", "tests/b.py", *mutation_worker.PYTEST_ARGUMENTS]
    assert [finder.module for finder in seen["finders"]] == ["pkg.mod"] and seen["forgot"] == "pkg.mod"
    assert sys.meta_path == before


def test_an_unmutated_job_installs_no_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    finders: list[int] = []
    monkeypatch.setattr(mutation_worker.pytest, "main", lambda _args: finders.append(len(sys.meta_path)) or 0)
    monkeypatch.setattr(mutation_worker, "forget_project", lambda _module: None)
    before = len(sys.meta_path)

    assert run_job("pkg.mod", SOURCE, None, ())[0] == 0
    assert finders == [before]


def test_a_failing_run_still_removes_the_finder(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(_arguments: list[str]) -> int:
        raise RuntimeError("pytest broke")

    monkeypatch.setattr(mutation_worker.pytest, "main", explode)
    monkeypatch.setattr(mutation_worker, "forget_project", lambda _module: None)
    before = list(sys.meta_path)

    with pytest.raises(RuntimeError, match="pytest broke"):
        run_job("pkg.mod", SOURCE, 0, ())
    assert sys.meta_path == before


def test_serve_preloads_says_ready_runs_each_job_and_stops_on_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mutation_worker, "run_job", lambda *job: (0, 0.5, f"ran {job[0]} {job[2]}"))
    mine, theirs = Pipe()
    worker = threading.Thread(target=serve, args=(theirs, ("json",)))
    worker.start()

    assert mine.recv() == READY
    mine.send(("m", "x = 1\n", 3, ("t.py",)))
    assert mine.recv() == (0, 0.5, "ran m 3")
    mine.send(None)
    worker.join(timeout=10)

    assert not worker.is_alive()
