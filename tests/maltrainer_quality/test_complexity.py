"""The complexity limit is a gate, so it must reject complexity 11 in any function.

A typo in the ruff invocation, or a C901 message ruff changed, would otherwise turn the limit off
while every other gate stays green.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from maltrainer_quality import complexity
from maltrainer_quality.complexity import MAX_COMPLEXITY

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def a_function_with_complexity(value: int, name: str = "classify") -> str:
    branches = "".join(f"    if flags[{index}]:\n        total += {index}\n" for index in range(value - 1))
    return f"def {name}(flags):\n    total = 0\n{branches}    return total\n"


def a_project(root: Path, **sources: str) -> Path:
    shutil.copy(PYPROJECT, root / "pyproject.toml")
    for name, source in sources.items():
        (root / f"{name}.py").write_text(source, encoding="utf-8")
    return root


def run_gate(root: Path, *arguments: str) -> int:
    return complexity.main(["--root", str(root), *arguments])


def test_complexity_eleven_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = a_project(tmp_path, subject=a_function_with_complexity(MAX_COMPLEXITY + 1))

    assert run_gate(root) == 1
    assert capsys.readouterr().out.startswith("subject.py::classify: 11 complexity (limit 10)")


def test_complexity_ten_is_accepted(tmp_path: Path) -> None:
    root = a_project(tmp_path, subject=a_function_with_complexity(MAX_COMPLEXITY))

    assert complexity.over_limit(root) == {}
    assert run_gate(root) == 0


def test_functions_sharing_a_name_in_one_file_are_numbered_in_file_order(tmp_path: Path) -> None:
    source = a_function_with_complexity(12) + "\n\n" + a_function_with_complexity(11)
    root = a_project(tmp_path, subject=source)

    assert complexity.over_limit(root) == {"subject.py::classify": 12, "subject.py::classify#2": 11}


def test_an_unexpected_message_fails_loudly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    finding = {"message": "something else", "filename": str(tmp_path / "x.py"), "location": {"row": 1}}
    monkeypatch.setattr(complexity, "findings", lambda _root: [finding])

    with pytest.raises(ValueError, match="unexpected C901 message"):
        complexity.over_limit(tmp_path)


def test_the_editor_setting_matches_the_gate() -> None:
    # A regex, not tomllib: the code supports Python 3.10, which has no tomllib.
    setting = re.search(r"^max-complexity = (\d+)$", PYPROJECT.read_text(encoding="utf-8"), re.MULTILINE)

    assert setting is not None
    assert int(setting[1]) == MAX_COMPLEXITY
