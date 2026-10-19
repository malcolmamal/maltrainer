"""
Training previews: render a few fixed prompts with the weights being trained, every ``every``
steps, without saving a checkpoint. Training pauses for the render and carries on.

    sample:
      every: 300                 # 0 = off
      at_start: true             # also render before the first step: the baseline to compare with
      rows: [raw, ema]           # a finetune with EMA: which weights to render (none = no previews)
      prompts:
        - "photo of feliciaday, a woman smiling"          # a plain line: enabled
        - prompt: "photo of a woman smiling"
          enabled: false                                   # defined, encoded, not rendered
      negative: ""               # the reference sampler's default
      steps: 30
      cfg: 3.0
      width: 1024
      height: 1024
      seed: 42                   # each prompt's seed: seed + its slot number, the same at every step

Files go to <output.path>/samples/: step000300_00.png per prompt, step000300.png with all of them
side by side, and prompts.txt naming each slot. A trainer can render several variants of the
weights (a finetune with EMA: the raw weights and the average): each is a row of the strip, and
its files are named step000300_00_raw.png / step000300_00_ema.png. ``rows`` picks which.

**Slots.** Each prompt keeps the number (and so the seed and file name) it got when first seen:
the config's prompts are 00, 01, ... in order, enabled or not, and a prompt added later takes the
next free number. Enabling, disabling, adding or removing prompts never renumbers the others.

**Changing them while training runs.** Every prompt in the file is encoded at the start, enabled
or not, so enabling one later is instant. Two ways in:
- ``update(config)``: the trainer re-reads the config file's sample section when the file is saved
- the preview menu (maltrainer/utils/preview_menu.py, key ``e``): ``set_every``, ``set_enabled``,
  ``set_row``
A prompt added to the file mid-run renders once the trainer has encoded it (``render``'s
``ready``); until then it is skipped.

The trainer brings the render function (prompt, seed) -> PIL image. ``captions`` lists what must
be encoded. ``current`` names the image being rendered ("step 300, image 2/4, ema") for the
trainer's progress bar. Each image is logged with its time, and one slower than ``slow_after``
seconds (24 s is normal at 1024 px on a 5090) is flagged: the usual cause is VRAM overflowing into
system RAM.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import Callable, ContextManager, Dict, List, Optional, Sequence, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

SETTINGS = ("every", "at_start", "negative", "steps", "cfg", "width", "height", "seed", "slow_after")
ROWS = ("raw", "ema")


def _entries(prompts) -> List[Tuple[str, bool]]:
    """``prompts`` items as (prompt, enabled): a string, or {prompt: ..., enabled: ...}."""
    out = []
    for item in prompts or []:
        if isinstance(item, dict):
            text = str(item.get("prompt", "")).strip()
            if text:
                out.append((text, bool(item.get("enabled", True))))
        elif str(item).strip():
            out.append((str(item), True))
    return out


class Previews:
    def __init__(self, config: Optional[dict], output_dir: Path):
        self.slots: Dict[str, int] = {}  # prompt -> its stable number (seed offset, file name)
        self.current = ""
        self.dir = Path(output_dir) / "samples"
        self._listed = False  # prompts.txt is rewritten once per process, then on every change
        self._apply(config or {})
        if self.every > 0 and not self.active:
            logger.warning("sample.every is set but no sample prompt is enabled: no previews")

    def _apply(self, config: dict) -> None:
        self.every = int(config.get("every", 0) or 0)
        entries = _entries(config.get("prompts"))
        self.prompts: List[str] = [p for p, _ in entries]  # every defined prompt, in file order
        self.enabled_prompts = {p for p, on in entries if on}
        self.at_start = bool(config.get("at_start", True))
        self.negative = str(config.get("negative", "") or "")
        self.steps = int(config.get("steps", 30))
        self.cfg = float(config.get("cfg", 3.0))
        self.width = int(config.get("width", 1024))
        self.height = int(config.get("height", 1024))
        self.seed = int(config.get("seed", 42))
        self.slow_after = float(config.get("slow_after", 60))
        rows = config.get("rows", list(ROWS))
        self.rows = {row: row in [str(r).lower() for r in (rows or [])] for row in ROWS}
        for prompt in self.prompts:
            if prompt not in self.slots:
                self.slots[prompt] = max(self.slots.values(), default=-1) + 1

    @property
    def active(self) -> List[str]:
        """The enabled prompts, in file order."""
        return [p for p in self.prompts if p in self.enabled_prompts]

    @property
    def enabled(self) -> bool:
        return self.every > 0 and bool(self.active)

    # ------------------------------------------------------------------ changes while running

    def update(self, config: Optional[dict]) -> List[str]:
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

    # ------------------------------------------------------------------ rendering

    @property
    def captions(self) -> List[str]:
        """Everything a render may need encoded: every defined prompt (enabled or not, so enabling
        one later is instant) and, with CFG, the negative."""
        if not self.prompts:
            return []
        return self.prompts + ([self.negative] if self.cfg != 1.0 else [])

    def due(self, step: int, total: int) -> bool:
        if not self.enabled:
            return False
        return (step == 0 and self.at_start) or (step > 0 and (step % self.every == 0 or step == total))

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

    def render(
        self,
        step: int,
        render_one: Callable[[str, int], Image.Image],
        variants: Sequence[Tuple[str, Callable[[], ContextManager]]] = (("", contextlib.nullcontext),),
        ready: Callable[[str], bool] = lambda prompt: True,
    ) -> float:
        """Render every enabled, ready prompt for ``step``, once per variant (label, context that
        puts those weights in place; labelled variants only when their row is on); returns the
        seconds it took."""
        self.dir.mkdir(parents=True, exist_ok=True)
        if not self._listed:
            self._write_listing()
            self._listed = True
        variants = [(label, weights) for label, weights in variants if not label or self.rows.get(label, True)]
        prompts = [p for p in self.active if ready(p)]
        waiting = [p for p in self.active if p not in prompts]
        if waiting:
            logger.info(
                "[sample] step %d: %d new prompt(s) still being encoded, skipped this round: %s",
                step,
                len(waiting),
                ", ".join(f"[{self.slots[p]:02d}]" for p in waiting),
            )
        if not prompts or not variants:
            return 0.0
        started = time.time()
        rows = []
        count = len(variants) * len(prompts)
        k = 0
        for label, weights in variants:
            images = []
            with weights():
                for prompt in prompts:
                    slot = self.slots[prompt]
                    k += 1
                    self.current = f"step {step}, image {k}/{count}" + (f", {label}" if label else "")
                    one = time.time()
                    image = render_one(prompt, self.seed + slot)
                    image.save(self.dir / f"step{step:06d}_{slot:02d}{'_' + label if label else ''}.png")
                    images.append(image)
                    took = time.time() - one
                    logger.info("[sample] %s: %.0fs", self.current, took)
                    if took > self.slow_after:
                        logger.warning(
                            "[sample] %s took %.0fs (over %.0fs): the GPU is probably out of VRAM and spilling into "
                            "system RAM. Close other GPU programs; NVIDIA Control Panel's 'CUDA - Sysmem Fallback "
                            "Policy: Prefer No Sysmem Fallback' turns this into an out-of-memory error instead",
                            self.current,
                            took,
                            self.slow_after,
                        )
            rows.append(images)
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
        strip.save(self.dir / f"step{step:06d}.png")
        elapsed = time.time() - started
        labels = ", ".join(label for label, _ in variants if label)
        logger.info(
            "[sample] step %d: %d preview(s)%s in %.0fs -> %s",
            step,
            sum(len(r) for r in rows),
            f" ({labels}, one row each)" if labels else "",
            elapsed,
            self.dir / f"step{step:06d}.png",
        )
        return elapsed
