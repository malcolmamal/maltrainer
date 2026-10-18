"""
Single-key commands while training runs in a console:

- ``s`` saves a checkpoint now; training continues
- ``q`` saves a checkpoint and stops (only once the save succeeded)
- ``p`` pauses training until ``p`` is pressed again; ``s``, ``q`` and ``h`` still work while paused
- ``h`` lists the keys

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
import time
from typing import Callable, List

logger = logging.getLogger(__name__)

SAVE, QUIT, PAUSE, HELP = "s", "q", "p", "h"
HELP_TEXT = {
    SAVE: "save a checkpoint now (training continues)",
    QUIT: "save a checkpoint and stop",
    PAUSE: "pause training; press p again to resume (s, q and h still work while paused)",
    HELP: "list these keys",
}


class KeyCommands:
    def __init__(self, keys: str = SAVE + QUIT + PAUSE + HELP):
        self.keys = set(keys)
        self._restore = None
        self.paused_seconds = 0.0  # time spent paused since take_paused_seconds() last read it
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
            logger.info("Keys: %s", ", ".join(f"[{k}] {HELP_TEXT[k].split(';')[0].split(' (')[0]}"
                                              for k in (SAVE, QUIT, PAUSE, HELP) if k in self.keys))

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

    def show_help(self) -> None:
        logger.info("Keys (act at the next step):")
        for key in (SAVE, QUIT, PAUSE, HELP):
            if key in self.keys:
                logger.info("  [%s] %s", key, HELP_TEXT[key])

    def take_paused_seconds(self) -> float:
        """Seconds spent paused since the last call (for the monitor's s/step)."""
        seconds, self.paused_seconds = self.paused_seconds, 0.0
        return seconds

    def _save_keys(self, pressed: List[str], step: int, save: Callable[[int], None]) -> bool:
        """``s`` / ``q`` among ``pressed``: at most one save, so ``s`` then ``q`` does not write the
        same step twice. True when the run should stop."""
        if SAVE not in pressed and QUIT not in pressed:
            return False
        stop = QUIT in pressed
        logger.info("Key [%s]: saving step %d%s", "q" if stop else "s", step, " and stopping" if stop else "")
        saved = save(step)
        if stop and saved is False:
            # never stop on a save that did not happen (e.g. the disk was full): the run is all there is
            logger.error("Key [q]: step %d was not saved, so training continues; free space and press q again", step)
            return False
        return stop

    def _pause(self, step: int, save: Callable[[int], None]) -> bool:
        """Wait at this step until ``p`` again; True if ``q`` (with a successful save) ends the run."""
        logger.info("Paused at step %d: [p] resume, [s] save, [q] save and stop, [h] help", step)
        started = time.time()
        try:
            while True:
                pressed = self.poll()
                if HELP in pressed:
                    self.show_help()
                if self._save_keys(pressed, step, save):
                    return True
                if PAUSE in pressed:
                    logger.info("Resumed at step %d after %.0f s paused", step, time.time() - started)
                    return False
                time.sleep(0.2)
        finally:
            self.paused_seconds += time.time() - started

    def handle(self, step: int, save: Callable[[int], None]) -> bool:
        """Act on pending keys at a step boundary; True when the run should stop."""
        pressed = self.poll()
        if not pressed:
            return False
        if HELP in pressed:
            self.show_help()
        if self._save_keys(pressed, step, save):
            return True
        if PAUSE in pressed:
            return self._pause(step, save)
        return False

    def close(self) -> None:
        if self._restore is not None:
            self._restore()
            self._restore = None
