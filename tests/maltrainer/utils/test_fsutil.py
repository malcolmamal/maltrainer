from __future__ import annotations

import logging
from pathlib import Path

import pytest

from maltrainer.utils.fsutil import rename_with_retry, replace_directory


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


def _folder(path: Path, content: str) -> Path:
    path.mkdir(parents=True)
    (path / "file.txt").write_text(content, encoding="utf-8")
    return path


def test_a_finished_folder_takes_the_place_of_the_old_one_and_nothing_is_left_over(tmp_path: Path) -> None:
    target = _folder(tmp_path / "state", "old")
    finished = _folder(tmp_path / "state.tmp", "new")

    replace_directory(finished, target)

    assert (target / "file.txt").read_text(encoding="utf-8") == "new"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state"]


def test_a_first_save_has_no_old_folder_to_move(tmp_path: Path) -> None:
    finished = _folder(tmp_path / "state.tmp", "new")

    replace_directory(finished, tmp_path / "state")

    assert (tmp_path / "state" / "file.txt").read_text(encoding="utf-8") == "new"


def test_a_leftover_old_folder_from_a_crashed_save_does_not_block_the_next(tmp_path: Path) -> None:
    _folder(tmp_path / "state.old", "stale")
    target = _folder(tmp_path / "state", "old")
    finished = _folder(tmp_path / "state.tmp", "new")

    replace_directory(finished, target)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["state"]
    assert (target / "file.txt").read_text(encoding="utf-8") == "new"


def test_a_refused_rename_while_swapping_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = _folder(tmp_path / "state", "old")
    finished = _folder(tmp_path / "state.tmp", "new")
    calls = refuse_first(monkeypatch, 1)
    monkeypatch.setattr("maltrainer.utils.fsutil.time.sleep", lambda _seconds: None)

    replace_directory(finished, target)

    assert (target / "file.txt").read_text(encoding="utf-8") == "new"
    assert len(calls) == 3  # the move aside refused, the move aside again, the move in
