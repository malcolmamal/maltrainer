from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from maltrainer_quality import mutation
from maltrainer_quality.mutation import Report, describe, module_name, run_target, to_json
from maltrainer_quality.mutation_ops import Site
from maltrainer_quality.mutation_pool import Outcome, Pool
from maltrainer_quality.mutation_select import changed_files, select_changed
from maltrainer_quality.mutation_targets import TARGETS, Target

MODULE = "def add(a, b):\n    return a + b\n\n\ndef spin(n):\n    while n > 0:\n        n -= 1\n    return n\n"
TESTS = "from mutsample.mod import add, spin\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
STRONG = TESTS + "\n\ndef test_spin():\n    assert spin(3) == 0\n    assert spin(-5) == -5\n"


def a_sample_package(root: Path, tests: str, monkeypatch: pytest.MonkeyPatch) -> Target:
    package = root / "mutsample"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mod.py").write_text(MODULE, encoding="utf-8")
    (package / "test_mod.py").write_text(tests, encoding="utf-8")
    monkeypatch.syspath_prepend(str(root))
    return Target("mutsample/mod.py", ("mutsample/test_mod.py",), 80.0)


def test_a_mutant_that_the_tests_miss_survives_and_the_others_are_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = a_sample_package(tmp_path, TESTS, monkeypatch)

    with Pool(1, preload=()) as pool:
        report = run_target(pool, target, tmp_path)

    assert [site.line for site in report.survivors] == [6, 6, 7, 7, 8]
    assert report.total == 7
    assert not report.passed
    assert "BELOW FLOOR" in describe(report)


def test_stronger_tests_kill_the_loop_mutants_including_one_that_never_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = a_sample_package(tmp_path, STRONG, monkeypatch)
    monkeypatch.setattr(mutation, "MIN_TIMEOUT", 5.0)
    monkeypatch.setattr(mutation, "TIMEOUT_FACTOR", 0.0)

    with Pool(2, preload=()) as pool:
        report = run_target(pool, target, tmp_path)

    assert report.survivors == ()
    assert report.passed
    assert report.score == 100.0


def test_failing_unmutated_tests_stop_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = a_sample_package(tmp_path, TESTS.replace("== 3", "== 4"), monkeypatch)

    with Pool(1, preload=()) as pool, pytest.raises(SystemExit, match="unmutated tests fail"):
        run_target(pool, target, tmp_path)


def test_a_worker_that_dies_is_replaced_and_the_mutant_counts_as_killed() -> None:
    with Pool(1, preload=()) as pool:
        worker = pool.workers[0]
        worker.process.kill()
        worker.process.join()
        died = worker.run(("m", "x = 1\n", None, ()), 5.0)
        again = worker.run(("m", "x = 1\n", None, ("no_such_test_file.py",)), 60.0)

    assert not died.survived
    assert again.code != 0


def test_module_names_come_from_paths() -> None:
    assert module_name("maltrainer/training/sigma.py") == "maltrainer.training.sigma"


def test_the_report_lists_survivors_and_to_json_round_trips() -> None:
    target = Target("pkg/m.py", ("tests/t.py",), 90.0)
    report = Report(target, 4, (Site(2, 7, "a + b  ->  a - b"),), 1.26)

    assert report.score == 75.0
    assert describe(report).splitlines() == [
        "pkg/m.py: 3/4 killed, 75.0% (floor 90.0) BELOW FLOOR",
        "    survived pkg/m.py:7: a + b  ->  a - b",
    ]
    assert json.loads(to_json([report])) == {
        "pkg/m.py": {
            "total": 4,
            "survived": [{"line": 7, "change": "a + b  ->  a - b"}],
            "score": 75.0,
            "floor": 90.0,
            "seconds": 1.3,
        }
    }
    assert Report(target, 0, (), 0.0).score == 100.0
    assert Outcome(0, 1.0, "").survived
    assert not Outcome(0, 1.0, "", timed_out=True).survived


def test_every_target_exists_and_so_do_its_tests_and_floors_are_sane() -> None:
    root = Path(__file__).resolve().parents[2]

    assert len({target.path for target in TARGETS}) == len(TARGETS)
    for target in TARGETS:
        assert (root / target.path).is_file(), target.path
        assert target.tests
        assert all((root / test).is_file() for test in target.tests), target.path
        assert 0 < target.floor <= 100


def test_main_lists_targets_and_rejects_unknown_modules(capsys: pytest.CaptureFixture[str]) -> None:
    assert mutation.main(["--list"]) == 0
    assert TARGETS[0].path in capsys.readouterr().out
    with pytest.raises(SystemExit, match="Unknown"):
        mutation.main(["nope.py"])
    with pytest.raises(SystemExit, match="name target modules"):
        mutation.main([])


def test_changed_selects_targets_by_module_or_by_test_file() -> None:
    first = TARGETS[0]

    assert select_changed({first.path}) == [t for t in TARGETS if first.path in {t.path, *t.tests}]
    assert first in select_changed({first.tests[0]})
    assert select_changed({"README.md"}) == []


def test_changed_files_are_those_since_the_fork_point(tmp_path: Path) -> None:
    def git(*arguments: str) -> None:
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *arguments], cwd=tmp_path, check=True)

    git("init", "-q", "-b", "main")
    (tmp_path / "old.py").write_text("x = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "one")
    git("checkout", "-q", "-b", "work")
    (tmp_path / "committed.py").write_text("x = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-q", "-m", "two")
    (tmp_path / "old.py").write_text("x = 2\n", encoding="utf-8")
    (tmp_path / "untracked.py").write_text("x = 1\n", encoding="utf-8")

    assert changed_files(tmp_path, "main") == {"committed.py", "old.py", "untracked.py"}
    with pytest.raises(SystemExit, match="git merge-base"):
        changed_files(tmp_path, "no-such-branch")
