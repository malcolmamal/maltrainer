from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from maltrainer.utils import lora_metadata
from maltrainer.utils.lora_metadata import git_commit


def _git(root: Path, *args: str) -> str:
    command = [
        "git",
        "-C",
        str(root),
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
        "-c",
        "commit.gpgsign=false",
    ]
    return subprocess.run([*command, *args], check=True, capture_output=True, text=True).stdout.strip()


def _repo_with_a_commit(root: Path) -> str:
    _git(root, "init", "-q")
    (root / "train.py").write_text("print('hi')\n", encoding="utf-8")
    _git(root, "add", "train.py")
    _git(root, "commit", "-q", "-m", "first")
    return _git(root, "rev-parse", "--short", "HEAD")


def test_a_clean_checkout_records_its_short_commit(tmp_path: Path) -> None:
    expected = _repo_with_a_commit(tmp_path)

    assert git_commit(tmp_path) == expected


def test_a_checkout_with_local_changes_is_marked_dirty(tmp_path: Path) -> None:
    expected = _repo_with_a_commit(tmp_path)
    (tmp_path / "train.py").write_text("print('changed')\n", encoding="utf-8")

    assert git_commit(tmp_path) == f"{expected}-dirty"


def test_a_repository_without_commits_has_no_commit(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")

    assert git_commit(tmp_path) is None


class _FakeRun:
    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        self.commands.append(command)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return subprocess.CompletedProcess(command, 0, stdout=outcome)


@pytest.mark.parametrize("error", [FileNotFoundError("git"), subprocess.TimeoutExpired("git", 10)])
def test_no_commit_is_recorded_when_git_cannot_run(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(lora_metadata.subprocess, "run", _FakeRun(error))

    assert git_commit(Path("repo")) is None


def test_an_empty_answer_from_git_records_no_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lora_metadata.subprocess, "run", _FakeRun("  \n"))

    assert git_commit(Path("repo")) is None


def test_the_commit_is_kept_when_the_status_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRun("abc1234\n", OSError("status"))
    monkeypatch.setattr(lora_metadata.subprocess, "run", fake)

    assert git_commit(Path("repo")) == "abc1234"
    assert fake.commands == [
        ["git", "-C", "repo", "rev-parse", "--short", "HEAD"],
        ["git", "-C", "repo", "status", "--porcelain"],
    ]


def test_without_a_root_the_trainer_checkout_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRun("abc1234\n", "")
    monkeypatch.setattr(lora_metadata.subprocess, "run", fake)

    assert git_commit() == "abc1234"
    assert fake.commands[0][2] == str(Path(lora_metadata.__file__).resolve().parents[2])
