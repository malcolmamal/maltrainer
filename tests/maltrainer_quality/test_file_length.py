from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from maltrainer_quality import file_length, ratchet
from maltrainer_quality.file_length import MAX_LINES


def a_repository(root: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def write_lines(path: Path, count: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x = 1\n" * count, encoding="utf-8")
    return path


def run_gate(root: Path, *arguments: str) -> int:
    return file_length.main(["--root", str(root), *arguments])


def test_a_new_file_over_the_limit_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = a_repository(tmp_path)
    write_lines(root / "pkg" / "big.py", MAX_LINES + 1)

    assert run_gate(root) == 1
    assert capsys.readouterr().out.startswith(f"pkg/big.py: {MAX_LINES + 1} lines (limit {MAX_LINES})")


def test_a_file_at_the_limit_passes(tmp_path: Path) -> None:
    root = a_repository(tmp_path)
    write_lines(root / "ok.py", MAX_LINES)

    assert run_gate(root) == 0


def test_only_python_and_shell_files_count(tmp_path: Path) -> None:
    root = a_repository(tmp_path)
    write_lines(root / "run.sh", MAX_LINES + 1)
    write_lines(root / "notes.md", MAX_LINES + 1)

    assert file_length.over_limit(root) == {"run.sh": MAX_LINES + 1}


def test_ignored_files_do_not_count(tmp_path: Path) -> None:
    root = a_repository(tmp_path)
    (root / ".gitignore").write_text("venv/\n", encoding="utf-8")
    write_lines(root / "venv" / "big.py", MAX_LINES + 1)

    assert file_length.over_limit(root) == {}


def test_a_tracked_file_deleted_from_the_working_tree_is_skipped(tmp_path: Path) -> None:
    root = a_repository(tmp_path)
    gone = write_lines(root / "gone.py", MAX_LINES + 1)
    subprocess.run(["git", "add", "gone.py"], cwd=root, check=True)
    gone.unlink()

    assert file_length.candidates(root) == ()


def test_a_baselined_file_may_stay_but_not_grow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = a_repository(tmp_path)
    big = write_lines(root / "big.py", 250)
    ratchet.save(root / file_length.BASELINE, {"big.py": 250})

    assert run_gate(root) == 0
    write_lines(big, 251)
    assert run_gate(root) == 1
    assert "big.py: 251 lines, grew past its baseline of 250" in capsys.readouterr().out


def test_a_shrunk_file_fails_until_its_baseline_is_tightened(tmp_path: Path) -> None:
    root = a_repository(tmp_path)
    write_lines(root / "big.py", 240)
    write_lines(root / "split.py", 150)
    ratchet.save(root / file_length.BASELINE, {"big.py": 250, "split.py": 250})

    assert run_gate(root) == 1
    assert run_gate(root, "--tighten") == 0
    assert ratchet.load(root / file_length.BASELINE) == {"big.py": 240}
    assert run_gate(root) == 0
