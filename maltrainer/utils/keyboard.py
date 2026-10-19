"""
Single-key commands while training runs in a console (training waits while ``e`` / ``g`` work):

- ``s`` saves a checkpoint now; ``q`` saves and stops (only once the save succeeded)
- ``p`` pauses until ``p`` again; ``s``, ``q`` and ``h`` still work while paused
- ``e`` opens the trainer's menu, if it has one (MOSS-Image: the preview settings)
- ``g`` renders a gallery, if it has one (MOSS-Image finetune: maltrainer/utils/gallery.py)
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
from typing import Callable, List, Optional

from maltrainer.utils.console_keys import read_key as read_key  # re-exported: the trainers import it from here

logger = logging.getLogger(__name__)

SAVE, QUIT, PAUSE, MENU, GALLERY, HELP = "s", "q", "p", "e", "g", "h"
KEY_ORDER = (SAVE, QUIT, PAUSE, MENU, GALLERY, HELP)
HELP_TEXT = {
    SAVE: "save a checkpoint now (training continues)",
    QUIT: "save a checkpoint and stop",
    PAUSE: "pause training; press p again to resume (s, q and h still work while paused)",
    MENU: "preview settings: prompts on/off, the interval, the raw/EMA rows (training waits)",
    GALLERY: "gallery: render a few trained people at random with generic prompts (training waits)",
    HELP: "list these keys",
}


class KeyCommands:
    def __init__(
        self,
        keys: str = SAVE + QUIT + PAUSE + HELP,
        menu: Optional[Callable[[], None]] = None,
        gallery: Optional[Callable[[], None]] = None,
    ):
        """``menu``: opened by ``e``; ``gallery``: run by ``g`` (each key exists only when a trainer
        offers it)."""
        self.menu = menu
        self.gallery = gallery
        self.keys = set(keys) | ({MENU} if menu is not None else set()) | ({GALLERY} if gallery is not None else set())
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
            except (ImportError, OSError, ValueError) as e:
                logger.info("Key commands unavailable: %s", e)
                self.enabled = False
        if self.enabled:
            shown = [k for k in KEY_ORDER if k in self.keys]
            logger.info(
                "Keys: %s", ", ".join(f"[{k}] {HELP_TEXT[k].split(';')[0].split(' (')[0].split(':')[0]}" for k in shown)
            )

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
        for key in (k for k in KEY_ORDER if k in self.keys):
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

    def _run_extras(self, pressed: List[str], count_time: bool = True) -> None:
        """Run the menu / gallery the keys asked for. A failure is logged and training goes on.
        ``count_time``: add the time to paused_seconds (not from a pause, whose own total covers it)."""
        for key, name, action in ((MENU, "menu", self.menu), (GALLERY, "gallery", self.gallery)):
            if key not in pressed or action is None:
                continue
            started = time.time()
            try:
                action()
            except Exception:  # a menu or gallery problem must not end the run
                logger.exception("The %s failed; training continues", name)
            finally:
                if count_time:
                    self.paused_seconds += time.time() - started

    def _pause(self, step: int, save: Callable[[int], None]) -> bool:
        """Wait at this step until ``p`` again; True if ``q`` (with a successful save) ends the run."""
        logger.info(
            "Paused at step %d: [p] resume, [s] save, [q] save and stop, [h] help%s%s",
            step,
            ", [e] preview settings" if self.menu is not None else "",
            ", [g] gallery" if self.gallery is not None else "",
        )
        started = time.time()
        try:
            while True:
                pressed = self.poll()
                if HELP in pressed:
                    self.show_help()
                self._run_extras(pressed, count_time=False)
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
        self._run_extras(pressed)
        if self._save_keys(pressed, step, save):
            return True
        if PAUSE in pressed:
            return self._pause(step, save)
        return False

    def close(self) -> None:
        if self._restore is not None:
            self._restore()
            self._restore = None
