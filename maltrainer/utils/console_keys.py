"""Blocking single-key reads for the console menus."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from typing import cast


def read_key() -> str:
    """One key press, blocking: "up", "down", "left", "right", "enter", "esc", "space", or the
    character itself (lower case)."""
    if os.name == "nt":
        msvcrt = import_module("msvcrt")  # by name: the stubs omit msvcrt off Windows
        char = msvcrt.getwch()
        if char in ("\x00", "\xe0"):
            return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(msvcrt.getwch(), "")
    else:
        import select

        char = sys.stdin.read(1)
        if char == "\x1b":
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                return "esc"
            if sys.stdin.read(1) != "[":
                return "esc"
            return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(sys.stdin.read(1), "")
    return cast(str, {"\r": "enter", "\n": "enter", "\x1b": "esc", " ": "space"}.get(char, char.lower()))
