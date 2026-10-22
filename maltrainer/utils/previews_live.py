"""What a running ``Previews`` can do besides rendering: take changes, and write what it holds to disk."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

SETTINGS = ("every", "at_start", "negative", "steps", "cfg", "width", "height", "seed", "slow_after")


class PreviewEdits:
    """``Previews``' settings changes (``update`` from the config file, ``set_*`` from the menu) and prompts.txt."""

    slots: dict[str, int]
    dir: Path
    every: int
    prompts: list[str]
    enabled_prompts: set[str]
    rows: dict[str, bool]
    seed: int
    negative: str
    steps: int
    cfg: float
    width: int
    height: int

    if TYPE_CHECKING:

        def _apply(self, config: dict[str, Any]) -> None: ...

    # ------------------------------------------------------------------ changes while running

    def update(self, config: dict[str, Any] | None) -> list[str]:
        """Apply a new ``sample`` section; returns a line per change (empty: nothing changed)."""
        before = {key: getattr(self, key) for key in SETTINGS}
        old_prompts, old_enabled, old_rows = list(self.prompts), set(self.enabled_prompts), dict(self.rows)
        self._apply(config or {})
        changes = [
            f"{key}: {before[key]!r} -> {getattr(self, key)!r}" for key in SETTINGS if getattr(self, key) != before[key]
        ]
        changes += [f"rows: {row} {'on' if on else 'off'}" for row, on in self.rows.items() if old_rows.get(row) != on]
        for p in self.prompts:
            if p not in old_prompts:
                changes.append(f"+ [{self.slots[p]:02d}] {p}" + ("" if p in self.enabled_prompts else " (disabled)"))
            elif (p in self.enabled_prompts) != (p in old_enabled):
                changes.append(f"{'enabled ' if p in self.enabled_prompts else 'disabled'} [{self.slots[p]:02d}] {p}")
        changes += [f"- [{self.slots[p]:02d}] {p}" for p in old_prompts if p not in self.prompts]
        if changes:
            self._write_listing()
        return changes

    def set_every(self, every: int) -> None:
        self.every = max(int(every), 0)

    def set_enabled(self, prompt: str, on: bool) -> None:
        if on:
            self.enabled_prompts.add(prompt)
        else:
            self.enabled_prompts.discard(prompt)

    def set_row(self, row: str, on: bool) -> None:
        self.rows[row] = bool(on)

    def _write_listing(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for prompt, slot in sorted(self.slots.items(), key=lambda item: item[1]):
            state = "on " if prompt in self.enabled_prompts else ("off" if prompt in self.prompts else "removed")
            lines.append(f"{slot:02d}  seed {self.seed + slot}  [{state}]  {prompt}\n")
        lines.append(
            f"every {self.every} steps, rows {', '.join(r for r, on in self.rows.items() if on) or 'none'}; "
            f"negative: {self.negative!r}, {self.steps} steps, CFG {self.cfg:g}, {self.width}x{self.height}\n"
        )
        (self.dir / "prompts.txt").write_text("".join(lines), encoding="utf-8")


def contact_strip(rows: list[list[Image.Image]]) -> Image.Image:
    """The rows of images pasted one under the other, each row left to right."""
    width = max(sum(im.width for im in row) for row in rows)
    height = sum(max(im.height for im in row) for row in rows)
    strip = Image.new("RGB", (width, height))
    y = 0
    for row in rows:
        x = 0
        for image in row:
            strip.paste(image, (x, y))
            x += image.width
        y += max(im.height for im in row)
    return strip
