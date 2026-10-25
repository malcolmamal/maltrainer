"""Keys a trainer registers itself (``extra``): they are read, listed, run at a step boundary and in a pause."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from maltrainer.utils import keyboard
from maltrainer.utils.keyboard import KeyCommands
from tests.maltrainer.utils.keyboard_fakes import FakeClock, FakeMsvcrt, FakeTty


def _commands(monkeypatch: pytest.MonkeyPatch, *batches: str, calls: list[str], fail: bool = False) -> KeyCommands:
    console = FakeMsvcrt(*batches)
    clock = FakeClock(console)
    monkeypatch.setattr(keyboard, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(keyboard, "time", SimpleNamespace(time=clock.time, sleep=clock.sleep))
    monkeypatch.setattr(sys, "stdin", FakeTty())
    monkeypatch.setitem(sys.modules, "msvcrt", console)

    def woman() -> None:
        calls.append("woman")
        if fail:
            raise RuntimeError("out of memory")

    extra = {"w": ('render "photo of a woman"', woman), "m": ("render a man", lambda: calls.append("man"))}
    return KeyCommands(extra=extra)


def test_an_extra_key_runs_its_action_at_the_step_boundary_and_leaves_training_going(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    keys = _commands(monkeypatch, "wm", calls=calls)

    stopped = keys.handle(5, lambda _step: None)

    assert calls == ["woman", "man"] and not stopped


def test_a_key_that_is_not_registered_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    keys = _commands(monkeypatch, "x", calls=calls)

    assert keys.poll() == [] and calls == []


def test_the_extra_keys_are_listed_after_the_standard_ones_with_their_own_help(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    keys = _commands(monkeypatch, "h", calls=[])
    caplog.clear()

    with caplog.at_level(logging.INFO):
        keys.handle(1, lambda _step: None)

    lines = [r.getMessage() for r in caplog.records]
    assert lines[1:] == [
        "  [s] save a checkpoint now (training continues)",
        "  [q] save a checkpoint and stop",
        "  [p] pause training; press p again to resume (s, q and h still work while paused)",
        "  [h] list these keys",
        '  [w] render "photo of a woman"',
        "  [m] render a man",
    ]


def test_the_startup_line_names_the_extra_keys(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        _commands(monkeypatch, "", calls=[])

    assert '[w] render "photo of a woman"' in caplog.text and "[m] render a man" in caplog.text


def test_an_extra_key_may_take_the_gallery_key_when_the_trainer_has_no_gallery(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    console = FakeMsvcrt("g")
    monkeypatch.setattr(keyboard, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(sys, "stdin", FakeTty())
    monkeypatch.setitem(sys.modules, "msvcrt", console)
    keys = KeyCommands(extra={"g": ("render everything", lambda: calls.append("all"))})

    keys.handle(1, lambda _step: None)

    assert calls == ["all"] and keyboard.HELP_TEXT["g"].startswith("gallery")  # the module's table is untouched


@pytest.mark.parametrize("key", ["s", "q", "p", "h", "ab", ""])
def test_a_key_that_is_a_command_or_not_one_character_is_refused(key: str) -> None:
    with pytest.raises(ValueError, match="cannot be registered"):
        KeyCommands(extra={key: ("x", lambda: None)})


def test_a_key_the_trainers_gallery_already_uses_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be registered"):
        KeyCommands(gallery=lambda: None, extra={"g": ("x", lambda: None)})


def test_a_failing_action_is_logged_and_training_continues(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[str] = []
    keys = _commands(monkeypatch, "w", calls=calls, fail=True)

    with caplog.at_level(logging.ERROR):
        stopped = keys.handle(1, lambda _step: None)

    assert not stopped and "The [w] action failed; training continues" in caplog.text


def test_an_extra_key_works_while_paused_and_its_time_is_not_counted_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    keys = _commands(monkeypatch, "p", "w", "p", calls=calls)

    keys.handle(3, lambda _step: None)

    assert calls == ["woman"] and keys.paused_seconds == pytest.approx(0.4)
