from __future__ import annotations

import logging
from pathlib import Path

import pytest

from maltrainer.utils.fsutil import rename_with_retry


def refuse_first(monkeypatch: pytest.MonkeyPatch, times: int) -> list[int]:
    """Make ``Path.rename`` raise a permission error for its first ``times`` calls; the list counts calls."""
    real = Path.rename
    calls: list[int] = []

    def flaky(self: Path, target: Path) -> Path:
        calls.append(1)
        if len(calls) <= times:
            raise PermissionError(5, "Access is denied")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    return calls


def test_a_rename_that_works_is_done_once_without_waiting(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    waits: list[float] = []

    rename_with_retry(tmp_path / "a", tmp_path / "b", sleep=waits.append)

    assert (tmp_path / "b").is_dir()
    assert waits == []


def test_a_refused_rename_is_tried_again_with_growing_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "a").mkdir()
    calls = refuse_first(monkeypatch, 2)
    waits: list[float] = []

    with caplog.at_level(logging.WARNING):
        rename_with_retry(tmp_path / "a", tmp_path / "b", delay=0.5, sleep=waits.append)

    assert (tmp_path / "b").is_dir()
    assert (len(calls), waits) == (3, [0.5, 1.0])
    assert caplog.text.count("was refused") == 2


def test_a_rename_that_stays_refused_raises_after_the_last_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a").mkdir()
    calls = refuse_first(monkeypatch, 99)

    with pytest.raises(PermissionError):
        rename_with_retry(tmp_path / "a", tmp_path / "b", attempts=3, sleep=lambda _: None)

    assert len(calls) == 3


def test_an_error_that_is_not_a_permission_error_is_not_retried(tmp_path: Path) -> None:
    waits: list[float] = []

    with pytest.raises(FileNotFoundError):
        rename_with_retry(tmp_path / "missing", tmp_path / "b", sleep=waits.append)

    assert waits == []
