from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Callable, List, Optional

import pytest

from maltrainer.utils import keyboard
from maltrainer.utils.keyboard import KeyCommands
from tests.maltrainer.utils.keyboard_fakes import FakeClock, FakeMsvcrt, FakeTty


class _Saver:
    def __init__(self, result: Optional[bool] = None) -> None:
        self.result = result
        self.steps: List[int] = []

    def __call__(self, step: int) -> Optional[bool]:
        self.steps.append(step)
        return self.result


def _commands(
    monkeypatch: pytest.MonkeyPatch,
    *batches: str,
    menu: Optional[Callable[[], None]] = None,
    menu_seconds: Optional[float] = None,
) -> tuple[KeyCommands, FakeClock]:
    """Key commands on a Windows console whose first batch of keys is waiting; ``menu_seconds``
    gives them a menu that stays open that long."""
    console = FakeMsvcrt(*batches)
    clock = FakeClock(console)
    if menu_seconds is not None:
        menu = _slow_menu(clock, menu_seconds)
    monkeypatch.setattr(keyboard, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(keyboard, "time", SimpleNamespace(time=clock.time, sleep=clock.sleep))
    monkeypatch.setattr(sys, "stdin", FakeTty())
    monkeypatch.setitem(sys.modules, "msvcrt", console)
    return KeyCommands(menu=menu), clock


def _slow_menu(clock: FakeClock, seconds: float) -> Callable[[], None]:
    def menu() -> None:
        clock.now += seconds

    return menu


def test_no_key_means_training_carries_on(monkeypatch: pytest.MonkeyPatch) -> None:
    commands, _ = _commands(monkeypatch, "")
    save = _Saver()

    assert commands.handle(5, save) is False
    assert save.steps == []


def test_s_saves_and_training_continues(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    commands, _ = _commands(monkeypatch, "s")
    save = _Saver()

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.keyboard"):
        stop = commands.handle(5, save)

    assert stop is False
    assert save.steps == [5]
    assert caplog.messages[-1] == "Key [s]: saving step 5"


@pytest.mark.parametrize("keys", ["q", "sq"])
def test_q_saves_once_and_stops(monkeypatch: pytest.MonkeyPatch, keys: str) -> None:
    commands, _ = _commands(monkeypatch, keys)
    save = _Saver(result=True)

    assert commands.handle(7, save) is True
    assert save.steps == [7]


def test_q_does_not_stop_when_the_save_failed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    commands, _ = _commands(monkeypatch, "q")

    with caplog.at_level(logging.ERROR, logger="maltrainer.utils.keyboard"):
        stop = commands.handle(7, _Saver(result=False))

    assert stop is False
    assert caplog.messages == ["Key [q]: step 7 was not saved, so training continues; free space and press q again"]


def test_h_lists_the_keys(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    commands, _ = _commands(monkeypatch, "h")

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.keyboard"):
        commands.handle(1, _Saver())

    assert caplog.messages[-5:] == [
        "Keys (act at the next step):",
        "  [s] save a checkpoint now (training continues)",
        "  [q] save a checkpoint and stop",
        "  [p] pause training; press p again to resume (s, q and h still work while paused)",
        "  [h] list these keys",
    ]


def test_the_menu_time_counts_as_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    commands, _ = _commands(monkeypatch, "e", menu_seconds=4.0)

    assert commands.handle(3, _Saver()) is False
    assert commands.take_paused_seconds() == 4.0
    assert commands.take_paused_seconds() == 0.0


def test_e_does_nothing_without_a_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    commands, _ = _commands(monkeypatch, "e")

    assert commands.handle(3, _Saver()) is False
    assert commands.paused_seconds == 0.0


def test_a_failing_menu_does_not_end_the_run(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def broken() -> None:
        raise KeyError("prompt")

    commands, _ = _commands(monkeypatch, "e", menu=broken)

    with caplog.at_level(logging.ERROR, logger="maltrainer.utils.keyboard"):
        stop = commands.handle(3, _Saver())

    assert stop is False
    assert caplog.messages == ["The menu failed; training continues"]


def test_p_waits_until_p_again_and_counts_the_pause(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    commands, clock = _commands(monkeypatch, "p", "", "", "p")

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.keyboard"):
        stop = commands.handle(9, _Saver())

    assert stop is False
    assert clock.sleeps == [0.2, 0.2, 0.2]
    assert commands.take_paused_seconds() == pytest.approx(0.6)
    assert caplog.messages[-2:] == [
        "Paused at step 9: [p] resume, [s] save, [q] save and stop, [h] help",
        "Resumed at step 9 after 1 s paused",
    ]


def test_keys_still_work_while_paused_and_q_ends_the_pause(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    commands, _ = _commands(monkeypatch, "p", "s", "he", "q", menu_seconds=5.0)
    save = _Saver(result=True)

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.keyboard"):
        stop = commands.handle(9, save)

    assert stop is True
    assert save.steps == [9, 9]
    assert commands.take_paused_seconds() == pytest.approx(5.6)
    assert (
        "Paused at step 9: [p] resume, [s] save, [q] save and stop, [h] help, [e] preview settings" in caplog.messages
    )
    assert "Keys (act at the next step):" in caplog.messages


def test_closing_without_a_terminal_mode_is_harmless(monkeypatch: pytest.MonkeyPatch) -> None:
    commands, _ = _commands(monkeypatch, "")

    commands.close()

    assert commands.enabled is True
