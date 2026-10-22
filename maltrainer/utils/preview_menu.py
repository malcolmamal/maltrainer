"""
The preview settings menu (key ``e`` while training runs in a console): turn prompts on and off,
change how often previews render, pick the rows (raw / EMA). Training waits while it is open.

    Preview settings   ↑↓ move · space toggle · ←→ every ±100 · Enter apply · Esc cancel
    > Every:   300 steps   (0 = off)
      [x] raw weights
      [x] EMA average
      [x] 00  RAW photo of feliciaday, a woman smiling slightly, close-up with face …
      [ ] 01  RAW photo of keanureeves, a man with shoulder-length dark hair, close-…

Changes are made on a copy and applied with Enter; Esc leaves everything as it was. They last for
the run; set ``enabled:`` / ``every:`` / ``rows:`` in the config file to keep them (saving the file
applies its sample section too, see maltrainer/utils/previews.py). The menu is drawn in place with
ANSI escapes (Windows 10+ consoles and terminals support them).
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from typing import TextIO

from maltrainer.utils.previews import Previews

logger = logging.getLogger(__name__)

EVERY_STEP = 100
ROW_NAMES = {"raw": "raw weights", "ema": "EMA average"}
HEADER = "Preview settings   ↑↓ move · space toggle · ←→ every ±100 · Enter apply · Esc cancel"


class _Menu:
    """What the menu shows and edits: a copy of the settings, a cursor, and the diff against the originals."""

    def __init__(self, previews: Previews, row_labels: Sequence[str], ready: Callable[[str], bool]) -> None:
        self.previews = previews
        self.row_labels = row_labels
        self.ready = ready
        self.every = previews.every
        self.rows = {label: previews.rows.get(label, True) for label in row_labels}
        self.enabled = set(previews.enabled_prompts)
        self.items: list[tuple[str, str]] = (
            [("every", "")] + [("row", label) for label in row_labels] + [("prompt", p) for p in previews.prompts]
        )
        self.cursor = 0

    def _item_text(self, index: int) -> str:
        kind, value = self.items[index]
        mark = ">" if index == self.cursor else " "
        if kind == "every":
            return f"{mark} Every: {self.every:>5} steps   (0 = off)"
        if kind == "row":
            return f"{mark} [{'x' if self.rows[value] else ' '}] {ROW_NAMES.get(value, value)}"
        note = "" if self.ready(value) else "  (encoding...)"
        return f"{mark} [{'x' if value in self.enabled else ' '}] {self.previews.slots[value]:02d}  {value}{note}"

    def lines(self) -> list[str]:
        width = max(shutil.get_terminal_size((100, 20)).columns - 1, 40)
        out_lines = [HEADER]
        for i in range(len(self.items)):
            text = self._item_text(i)
            out_lines.append(text if len(text) <= width else text[: width - 1] + "…")
        if self.row_labels and not any(self.rows.values()):
            out_lines.append("  (both rows off: no previews)")
        elif self.every == 0:
            out_lines.append("  (every 0: no previews)")
        else:
            out_lines.append("")
        return out_lines

    def press(self, key: str) -> None:
        if key == "up":
            self.cursor = (self.cursor - 1) % len(self.items)
        elif key == "down":
            self.cursor = (self.cursor + 1) % len(self.items)
        elif key == "left":
            self.every = max(self.every - EVERY_STEP, 0)
        elif key == "right":
            self.every += EVERY_STEP
        elif key == "space":
            kind, value = self.items[self.cursor]
            if kind == "row":
                self.rows[value] = not self.rows[value]
            elif kind == "prompt":
                self.enabled.symmetric_difference_update({value})

    def apply(self) -> list[str]:
        """Write the edited settings back to the previews; returns a line per change."""
        previews = self.previews
        changes = []
        if self.every != previews.every:
            changes.append(f"every: {previews.every} -> {self.every}")
            previews.set_every(self.every)
        for label, on in self.rows.items():
            if previews.rows.get(label, True) != on:
                changes.append(f"rows: {label} {'on' if on else 'off'}")
                previews.set_row(label, on)
        for prompt in previews.prompts:
            on = prompt in self.enabled
            if (prompt in previews.enabled_prompts) != on:
                changes.append(f"{'enabled ' if on else 'disabled'} [{previews.slots[prompt]:02d}] {prompt}")
                previews.set_enabled(prompt, on)
        return changes


def run_menu(
    previews: Previews,
    row_labels: Sequence[str],
    read_key: Callable[[], str],
    ready: Callable[[str], bool] = lambda _prompt: True,
    out: TextIO | None = None,
) -> list[str]:
    """Show the menu until Enter (apply) or Esc (cancel); returns the applied changes. ``out``: the
    stdout of the moment it runs, not the one bound at import (a redirected console must be honoured)."""
    out = sys.stdout if out is None else out
    menu = _Menu(previews, row_labels, ready)
    drawn = 0
    if os.name == "nt":
        os.system("")  # turns on ANSI escape handling in the Windows console

    def draw() -> None:
        nonlocal drawn
        text = menu.lines()
        if drawn:
            out.write(f"\x1b[{drawn}F")  # back to the menu's first line
        for line in text:
            out.write("\x1b[2K" + line + "\n")
        out.flush()
        drawn = len(text)

    out.write("\n")
    draw()
    while True:
        key = read_key()
        if key == "esc":
            out.write("Preview settings: cancelled, nothing changed\n")
            out.flush()
            return []
        if key == "enter":
            break
        menu.press(key)
        draw()

    changes = menu.apply()
    if changes:
        previews._write_listing()
        logger.info(
            "[sample] preview settings changed (for this run; the config file is unchanged):\n  %s",
            "\n  ".join(changes),
        )
    else:
        out.write("Preview settings: no changes\n")
        out.flush()
    return changes
