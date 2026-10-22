"""The type check is a gate, so it must reject a type error in code anywhere in the tree.

An invocation that mypy ignores, or an output format it changed, would otherwise turn the check
off while every other gate stays green.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from maltrainer_quality import typing_gate

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
CLEAN = "def double(value: int) -> int:\n    return value * 2\n"
WRONG = "def double(value: int) -> int:\n    return str(value)\n"


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    shutil.copy(PYPROJECT, tmp_path / "pyproject.toml")
    (tmp_path / "package").mkdir()
    monkeypatch.setattr(typing_gate, "TARGETS", ("package",))
    return tmp_path


def write(root: Path, source: str, name: str = "subject") -> None:
    (root / "package" / f"{name}.py").write_text(source, encoding="utf-8")


def test_a_clean_tree_passes(project: Path) -> None:
    write(project, CLEAN)

    assert typing_gate.main(["--root", str(project)]) == 0


def test_a_type_error_is_rejected(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(project, WRONG)

    assert typing_gate.main(["--root", str(project)]) == 1
    assert capsys.readouterr().out.startswith("package/subject.py: 1 type errors")


def test_errors_are_counted_per_file_from_mypys_lines() -> None:
    output = (
        "package/a.py:3: error: Incompatible return value type  [return-value]\n"
        "package/a.py:9:5: error: Missing return statement  [return]\n"
        "package/a.py:9: note: a note is not an error\n"
        "package/b.py:1: error: bad  [misc]\n"
    )

    assert typing_gate.errors_per_file(output) == {"package/a.py": 2, "package/b.py": 1}


def test_a_mypy_that_cannot_run_fails_loudly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(typing_gate.importlib.util, "find_spec", lambda _name: None)

    with pytest.raises(RuntimeError, match="not installed"):
        typing_gate.mypy_output(tmp_path)


def test_a_mypy_that_crashes_is_not_taken_for_a_clean_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        typing_gate.subprocess,
        "run",
        lambda command, **_: subprocess.CompletedProcess(command, 2, stdout=b"boom"),
    )

    with pytest.raises(RuntimeError, match="mypy exited with 2:\nboom"):
        typing_gate.mypy_output(tmp_path)
