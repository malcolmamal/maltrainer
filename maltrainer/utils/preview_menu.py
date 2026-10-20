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
    every = previews.every
    rows = {label: previews.rows.get(label, True) for label in row_labels}
    enabled = set(previews.enabled_prompts)
    items: list[tuple[str, str]] = (
        [("every", "")] + [("row", label) for label in row_labels] + [("prompt", p) for p in previews.prompts]
    )
    cursor = 0
    drawn = 0
    if os.name == "nt":
        os.system("")  # turns on ANSI escape handling in the Windows console

    def lines() -> list[str]:
        width = max(shutil.get_terminal_size((100, 20)).columns - 1, 40)
        out_lines = [HEADER]
        for i, (kind, value) in enumerate(items):
            mark = ">" if i == cursor else " "
            if kind == "every":
                text = f"{mark} Every: {every:>5} steps   (0 = off)"
            elif kind == "row":
                text = f"{mark} [{'x' if rows[value] else ' '}] {ROW_NAMES.get(value, value)}"
            else:
                note = "" if ready(value) else "  (encoding...)"
                text = f"{mark} [{'x' if value in enabled else ' '}] {previews.slots[value]:02d}  {value}{note}"
            out_lines.append(text if len(text) <= width else text[: width - 1] + "…")
        if row_labels and not any(rows.values()):
            out_lines.append("  (both rows off: no previews)")
        elif every == 0:
            out_lines.append("  (every 0: no previews)")
        else:
            out_lines.append("")
        return out_lines

    def draw() -> None:
        nonlocal drawn
        text = lines()
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
        if key == "up":
            cursor = (cursor - 1) % len(items)
        elif key == "down":
            cursor = (cursor + 1) % len(items)
        elif key == "left":
            every = max(every - EVERY_STEP, 0)
        elif key == "right":
            every += EVERY_STEP
        elif key == "space":
            kind, value = items[cursor]
            if kind == "row":
                rows[value] = not rows[value]
            elif kind == "prompt":
                enabled.symmetric_difference_update({value})
        elif key == "esc":
            out.write("Preview settings: cancelled, nothing changed\n")
            out.flush()
            return []
        elif key == "enter":
            break
        draw()

    changes = []
    if every != previews.every:
        changes.append(f"every: {previews.every} -> {every}")
        previews.set_every(every)
    for label, on in rows.items():
        if previews.rows.get(label, True) != on:
            changes.append(f"rows: {label} {'on' if on else 'off'}")
            previews.set_row(label, on)
    for prompt in previews.prompts:
        on = prompt in enabled
        if (prompt in previews.enabled_prompts) != on:
            changes.append(f"{'enabled ' if on else 'disabled'} [{previews.slots[prompt]:02d}] {prompt}")
            previews.set_enabled(prompt, on)
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
