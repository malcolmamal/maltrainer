"""
Training previews: render a few fixed prompts with the weights being trained, every ``every``
steps, without saving a checkpoint. Training pauses for the render and carries on.

    sample:
      every: 50                  # 0 = off
      at_start: true             # also render before the first step: the baseline to compare with
      prompts: ["photo of feliciaday, a woman smiling", "photo of a woman"]
      negative: ""               # the reference sampler's default
      steps: 30
      cfg: 3.0
      width: 1024
      height: 1024
      seed: 42                   # prompt i uses seed + i, the same at every step

Files go to <output.path>/samples/: step000050_00.png per prompt, step000050.png with all of them
side by side, and prompts.txt naming each index. The same seed per prompt at every step makes the
pictures directly comparable over the run. A trainer can render several variants of the weights
(a finetune with EMA: the raw weights and the average): each is a row of the strip, and its
files are named step000050_00_raw.png / step000050_00_ema.png.

The trainer brings the render function (prompt, seed) -> PIL image; the prompts must be encoded
while the text encoder is loaded, so ``captions`` lists what to cache.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
import contextlib
from typing import Callable, ContextManager, List, Optional, Sequence, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class Previews:
    def __init__(self, config: Optional[dict], output_dir: Path):
        config = config or {}
        self.every = int(config.get("every", 0) or 0)
        self.prompts: List[str] = [str(p) for p in (config.get("prompts") or [])]
        self.enabled = self.every > 0 and bool(self.prompts)
        self.at_start = bool(config.get("at_start", True))
        self.negative = str(config.get("negative", "") or "")
        self.steps = int(config.get("steps", 30))
        self.cfg = float(config.get("cfg", 3.0))
        self.width = int(config.get("width", 1024))
        self.height = int(config.get("height", 1024))
        self.seed = int(config.get("seed", 42))
        self.dir = Path(output_dir) / "samples"
        if self.every > 0 and not self.prompts:
            logger.warning("sample.every is set but sample.prompts is empty: no previews")

    @property
    def captions(self) -> List[str]:
        """Everything a render needs encoded: the prompts and, with CFG, the negative."""
        if not self.enabled:
            return []
        return self.prompts + ([self.negative] if self.cfg != 1.0 else [])

    def due(self, step: int, total: int) -> bool:
        if not self.enabled:
            return False
        return (step == 0 and self.at_start) or (step > 0 and (step % self.every == 0 or step == total))

    def render(
        self,
        step: int,
        render_one: Callable[[str, int], Image.Image],
        variants: Sequence[Tuple[str, Callable[[], ContextManager]]] = (("", contextlib.nullcontext),),
    ) -> float:
        """Render every prompt for ``step``, once per variant (label, context that puts those
        weights in place); returns the seconds it took."""
        self.dir.mkdir(parents=True, exist_ok=True)
        listing = self.dir / "prompts.txt"
        if not listing.exists():
            listing.write_text(
                "".join(f"{i:02d}  seed {self.seed + i}  {p}\n" for i, p in enumerate(self.prompts))
                + f"negative: {self.negative!r}, {self.steps} steps, CFG {self.cfg:g}, {self.width}x{self.height}\n",
                encoding="utf-8",
            )
        started = time.time()
        rows = []
        for label, weights in variants:
            images = []
            with weights():
                for i, prompt in enumerate(self.prompts):
                    image = render_one(prompt, self.seed + i)
                    image.save(self.dir / f"step{step:06d}_{i:02d}{'_' + label if label else ''}.png")
                    images.append(image)
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
        logger.info("[sample] step %d: %d preview(s)%s in %.0fs -> %s", step, sum(len(r) for r in rows),
                    f" ({labels}, one row each)" if labels else "", elapsed, self.dir / f"step{step:06d}.png")
        return elapsed
