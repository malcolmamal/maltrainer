"""
Single-key commands while training runs in a console: ``s`` saves a checkpoint now, ``q`` saves
one and stops.

Keys are read without blocking at each optimizer step, so a press acts at the next step
boundary. Windows reads the console with ``msvcrt``; elsewhere stdin is put into cbreak mode for
the run and restored afterwards. Without a console (stdin is not a TTY, e.g. a trainer started
by the web UI), the listener is off.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
from typing import Callable, List

logger = logging.getLogger(__name__)

SAVE, QUIT = "s", "q"


class KeyCommands:
    def __init__(self, keys: str = SAVE + QUIT):
        self.keys = set(keys)
        self._restore = None
        try:
            self.enabled = sys.stdin is not None and sys.stdin.isatty()
        except (AttributeError, ValueError):
            self.enabled = False
        if self.enabled and os.name != "nt":
            try:
                import termios
                import tty

                fd = sys.stdin.fileno()
                saved = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                self._restore = lambda: termios.tcsetattr(fd, termios.TCSADRAIN, saved)
                atexit.register(self.close)
            except Exception as e:  # pragma: no cover - depends on the terminal
                logger.info("Key commands unavailable: %s", e)
                self.enabled = False
        if self.enabled:
            logger.info("Keys: [s] save a checkpoint now, [q] save and stop")

    def poll(self) -> List[str]:
        """The command keys pressed since the last poll, in order."""
        if not self.enabled:
            return []
        pressed: List[str] = []
        if os.name == "nt":
            import msvcrt

            while msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in ("\x00", "\xe0"):  # arrow / function keys come as two codes
                    msvcrt.getwch()
                    continue
                if char.lower() in self.keys:
                    pressed.append(char.lower())
        else:
            import select

            while select.select([sys.stdin], [], [], 0)[0]:
                char = sys.stdin.read(1)
                if not char:
                    break
                if char.lower() in self.keys:
                    pressed.append(char.lower())
        return pressed

    def handle(self, step: int, save: Callable[[int], None]) -> bool:
        """Act on pending keys at a step boundary; True when the run should stop.

        At most one save per poll, so ``s`` followed by ``q`` does not write the same step twice.
        """
        pressed = self.poll()
        if not pressed:
            return False
        stop = QUIT in pressed
        logger.info("Key [%s]: saving step %d%s", "q" if stop else "s", step, " and stopping" if stop else "")
        save(step)
        return stop

    def close(self) -> None:
        if self._restore is not None:
            self._restore()
            self._restore = None
