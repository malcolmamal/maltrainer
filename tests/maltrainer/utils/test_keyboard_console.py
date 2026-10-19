from __future__ import annotations

import io
import logging
import select
import sys
from types import SimpleNamespace
from typing import Any, List

import pytest

from maltrainer.utils import console_keys, keyboard
from maltrainer.utils.console_keys import read_key
from maltrainer.utils.keyboard import KeyCommands
from tests.maltrainer.utils.keyboard_fakes import FakeMsvcrt, FakeTty


def _on(monkeypatch: pytest.MonkeyPatch, os_name: str, stdin: Any) -> None:
    monkeypatch.setattr(keyboard, "os", SimpleNamespace(name=os_name))
    monkeypatch.setattr(console_keys, "os", SimpleNamespace(name=os_name))
    monkeypatch.setattr(sys, "stdin", stdin)


def _terminal(monkeypatch: pytest.MonkeyPatch, stdin: FakeTty, calls: List[tuple]) -> None:
    """A POSIX console: termios / tty record what they are asked, atexit records what it would run."""

    def tcgetattr(fd: int) -> list:
        calls.append(("get", fd))
        if stdin.getvalue() == "refuse":
            raise OSError("not a terminal")
        return ["saved"]

    termios = SimpleNamespace(TCSADRAIN=1, tcgetattr=tcgetattr)
    termios.tcsetattr = lambda fd, when, saved: calls.append(("set", fd, when, saved))
    monkeypatch.setitem(sys.modules, "termios", termios)
    monkeypatch.setitem(sys.modules, "tty", SimpleNamespace(setcbreak=lambda fd: calls.append(("cbreak", fd))))
    monkeypatch.setattr(keyboard, "atexit", SimpleNamespace(register=lambda f: calls.append(("atexit", f.__name__))))
    _on(monkeypatch, "posix", stdin)


def _select_while_unread(readers: List[io.StringIO], _w: list, _x: list, _timeout: float) -> tuple:
    stream = readers[0]
    return ([stream] if stream.tell() < len(stream.getvalue()) else [], [], [])


@pytest.mark.parametrize(
    ("chars", "expected"),
    [
        ("àH", "up"),
        ("\x00P", "down"),
        ("àK", "left"),
        ("àM", "right"),
        ("\x00;", ""),
        ("\r", "enter"),
        ("\x1b", "esc"),
        (" ", "space"),
        ("Q", "q"),
    ],
)
def test_windows_keys_are_named(monkeypatch: pytest.MonkeyPatch, chars: str, expected: str) -> None:
    _on(monkeypatch, "nt", None)
    monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt(chars))

    assert read_key() == expected


@pytest.mark.parametrize(
    ("chars", "expected"),
    [
        ("\x1b[A", "up"),
        ("\x1b[B", "down"),
        ("\x1b[C", "right"),
        ("\x1b[D", "left"),
        ("\x1b[Z", ""),
        ("\x1b", "esc"),
        ("\x1bO", "esc"),
        ("\n", "enter"),
        ("S", "s"),
    ],
)
def test_terminal_escape_sequences_are_named(monkeypatch: pytest.MonkeyPatch, chars: str, expected: str) -> None:
    _on(monkeypatch, "posix", io.StringIO(chars))
    monkeypatch.setattr(select, "select", _select_while_unread)

    assert read_key() == expected


@pytest.mark.parametrize("stdin", [None, io.StringIO("s")])
def test_without_a_console_the_keys_are_off(monkeypatch: pytest.MonkeyPatch, stdin: Any) -> None:
    _on(monkeypatch, "nt", stdin)

    commands = KeyCommands()

    assert commands.enabled is False
    assert commands.poll() == []


def test_a_closed_stdin_turns_the_keys_off(monkeypatch: pytest.MonkeyPatch) -> None:
    closed = io.StringIO()
    closed.close()
    _on(monkeypatch, "nt", closed)

    assert KeyCommands().enabled is False


def test_the_keys_are_announced_and_the_menu_key_exists_only_with_a_menu(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _on(monkeypatch, "nt", FakeTty())

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.keyboard"):
        plain = KeyCommands()
        with_menu = KeyCommands(keys="sq", menu=lambda: None)

    assert (plain.keys, with_menu.keys) == ({"s", "q", "p", "h"}, {"s", "q", "e"})
    assert caplog.messages == [
        "Keys: [s] save a checkpoint now, [q] save a checkpoint and stop, [p] pause training, [h] list these keys",
        "Keys: [s] save a checkpoint now, [q] save a checkpoint and stop, [e] preview settings",
    ]


def test_a_terminal_is_put_in_cbreak_mode_and_restored_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[tuple] = []
    _terminal(monkeypatch, FakeTty(), calls)

    commands = KeyCommands()
    commands.close()
    commands.close()

    assert commands.enabled is True
    assert calls == [("get", 7), ("cbreak", 7), ("atexit", "close"), ("set", 7, 1, ["saved"])]


def test_a_terminal_that_refuses_cbreak_mode_turns_the_keys_off(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[tuple] = []
    _terminal(monkeypatch, FakeTty("refuse"), calls)

    commands = KeyCommands()

    assert commands.enabled is False
    assert calls == [("get", 7)]


def test_windows_polling_keeps_command_keys_in_order_and_skips_arrows(monkeypatch: pytest.MonkeyPatch) -> None:
    _on(monkeypatch, "nt", FakeTty())
    monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt("S\xe0Hxq\x00Ph"))
    commands = KeyCommands()

    assert commands.poll() == ["s", "q", "h"]
    assert commands.poll() == []


def test_terminal_polling_reads_until_nothing_is_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    _terminal(monkeypatch, FakeTty("pzS"), [])
    monkeypatch.setattr(select, "select", _select_while_unread)
    commands = KeyCommands()

    assert commands.poll() == ["p", "s"]


def test_terminal_polling_stops_at_the_end_of_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _terminal(monkeypatch, FakeTty("q"), [])
    monkeypatch.setattr(select, "select", lambda readers, _w, _x, _t: (readers, [], []))
    commands = KeyCommands()

    assert commands.poll() == ["q"]
