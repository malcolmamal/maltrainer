from __future__ import annotations

import io
import logging
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from maltrainer.utils import preview_menu
from maltrainer.utils.preview_menu import HEADER, run_menu
from maltrainer.utils.previews import Previews


@pytest.fixture(autouse=True)
def console(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    commands: list[str] = []
    monkeypatch.setattr(preview_menu, "os", SimpleNamespace(name="posix", system=commands.append))
    monkeypatch.setattr(
        preview_menu, "shutil", SimpleNamespace(get_terminal_size=lambda _: os.terminal_size((121, 20)))
    )
    return commands


def _previews(tmp_path: Path) -> Previews:
    return Previews({"every": 100, "prompts": ["a", {"prompt": "b", "enabled": False}]}, tmp_path)


def _keys(*keys: str) -> Callable[[], str]:
    return iter(keys).__next__


def _screens(text: str) -> list[list[str]]:
    """Each drawing of the menu as its lines (a redraw starts by moving the cursor back up)."""
    return [[piece.split("\n")[0] for piece in frame.split("\x1b[2K")[1:]] for frame in re.split(r"\x1b\[\d+F", text)]


def _run(
    previews: Previews, keys: Sequence[str], row_labels: Sequence[str] = ("raw", "ema"), **kwargs: object
) -> tuple:
    out = io.StringIO()
    changes = run_menu(previews, row_labels, _keys(*keys), out=out, **kwargs)
    return changes, out.getvalue()


def test_the_menu_draws_every_setting_with_the_cursor_on_the_interval(tmp_path: Path) -> None:
    changes, text = _run(_previews(tmp_path), ["enter"])

    assert changes == []
    assert _screens(text) == [
        [
            HEADER,
            "> Every:   100 steps   (0 = off)",
            "  [x] raw weights",
            "  [x] EMA average",
            "  [x] 00  a",
            "  [ ] 01  b",
            "",
        ]
    ]
    assert text.endswith("Preview settings: no changes\n")


def test_enter_applies_the_changes_made_in_the_menu(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    previews = _previews(tmp_path)
    keys = ["right", "right", "left", "down", "space", "down", "down", "space", "down", "space", "enter"]

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.preview_menu"):
        changes, _ = _run(previews, keys)

    assert changes == ["every: 100 -> 200", "rows: raw off", "disabled [00] a", "enabled  [01] b"]
    assert (previews.every, previews.rows, previews.active) == (200, {"raw": False, "ema": True}, ["b"])
    assert (tmp_path / "samples" / "prompts.txt").exists()
    assert caplog.messages[0].endswith(
        "unchanged):\n  every: 100 -> 200\n  rows: raw off\n  disabled [00] a\n  enabled  [01] b"
    )


def test_escape_leaves_everything_as_it_was(tmp_path: Path) -> None:
    previews = _previews(tmp_path)

    changes, text = _run(previews, ["right", "down", "space", "esc"])

    assert changes == []
    assert (previews.every, previews.rows, previews.active) == (100, {"raw": True, "ema": True}, ["a"])
    assert text.endswith("Preview settings: cancelled, nothing changed\n")
    assert not (tmp_path / "samples").exists()


def test_the_cursor_wraps_around_both_ends(tmp_path: Path) -> None:
    previews = _previews(tmp_path)

    changes, text = _run(previews, ["up", "space", "down", "down", "space", "enter"])

    assert changes == ["rows: raw off", "enabled  [01] b"]
    assert _screens(text)[5][1:3] == ["  Every:   100 steps   (0 = off)", "> [ ] raw weights"]


def test_space_on_the_interval_changes_nothing(tmp_path: Path) -> None:
    assert _run(_previews(tmp_path), ["space", "x", "enter"])[0] == []


def test_the_menu_says_when_no_previews_would_render(tmp_path: Path) -> None:
    _, text = _run(_previews(tmp_path), ["down", "space", "down", "space", "left", "esc"])

    screens = _screens(text)
    assert screens[2][-1] == ""
    assert screens[4][-1] == "  (both rows off: no previews)"
    assert screens[5][-1] == "  (both rows off: no previews)"


def test_an_interval_of_zero_is_noted_and_cannot_go_below_zero(tmp_path: Path) -> None:
    previews = _previews(tmp_path)

    changes, text = _run(previews, ["left", "left", "enter"], row_labels=())

    assert _screens(text)[-1][1:] == [
        "> Every:     0 steps   (0 = off)",
        "  [x] 00  a",
        "  [ ] 01  b",
        "  (every 0: no previews)",
    ]
    assert changes == ["every: 100 -> 0"]


def test_prompts_not_yet_encoded_and_unknown_rows_are_labelled(tmp_path: Path) -> None:
    _, text = _run(_previews(tmp_path), ["enter"], row_labels=("other",), ready=lambda prompt: prompt == "a")

    assert _screens(text)[0][2:5] == ["  [x] other", "  [x] 00  a", "  [ ] 01  b  (encoding...)"]


def test_long_lines_are_cut_to_the_terminal_width(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preview_menu, "shutil", SimpleNamespace(get_terminal_size=lambda _: os.terminal_size((46, 20))))
    previews = Previews({"every": 100, "prompts": ["x" * 60]}, tmp_path)

    _, text = _run(previews, ["enter"], row_labels=())

    assert _screens(text)[0][2] == "  [x] 00  " + "x" * 34 + "…"


def test_the_windows_console_is_switched_to_ansi_first(tmp_path: Path, console: list[str]) -> None:
    preview_menu.os.name = "nt"

    _run(_previews(tmp_path), ["enter"])

    assert console == [""]
