"""The dead-code check is a gate, so it must reject an unused function.

vulture exits 0 when a path does not exist or a config key is misspelled, so a typo in pyproject.toml
would turn the check off while every other gate stays green.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def vulture(root: Path) -> subprocess.CompletedProcess[str]:
    shutil.copy(PYPROJECT, root / "pyproject.toml")
    return subprocess.run(
        [sys.executable, "-m", "vulture", "subject.py"], cwd=root, check=False, capture_output=True, text=True
    )


def test_an_unused_function_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "subject.py").write_text("def never_called():\n    return 1\n", encoding="utf-8")

    done = vulture(tmp_path)

    assert done.returncode == 3
    assert "unused function 'never_called'" in done.stdout


def test_a_used_function_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "subject.py").write_text("def helper():\n    return 1\n\n\nhelper()\n", encoding="utf-8")

    assert vulture(tmp_path).returncode == 0


def test_a_name_in_the_whitelist_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "subject.py").write_text("def epoch_order():\n    return 1\n", encoding="utf-8")

    assert vulture(tmp_path).returncode == 0
