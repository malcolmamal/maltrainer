"""
The gallery (key ``g``): while a multi-person finetune trains, render a handful of the trained
people at random, each with a generic prompt for their class word, to see how the whole set is
doing rather than the few fixed preview prompts.

    sample:
      gallery:
        enabled: true            # default: on when the finetune names its people (prior.classes)
        count: 8                 # people per press
        prompts:                 # per class word; {trigger} and {class} are filled in
          woman: ["photo of {trigger}, a woman smiling slightly, close-up with face visible, ..."]
          man: [...]
          default: [...]         # any other class word

Every person x prompt is encoded at the start with the training captions (and cached on disk like
them), so a press never waits for the text encoder. A press picks people and prompts with its own
random generator, never the training RNG, so a resumed run stays exact; each image gets its own
seed, written next to it. Files go to <output.path>/samples/gallery/: stepNNNNNN.png (all of them
in a grid), stepNNNNNN_<trigger>.png each, and stepNNNNNN.txt (trigger, prompt, seed).
"""

from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path
from typing import Callable, ContextManager, Dict, List

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS: Dict[str, List[str]] = {
    "woman": [
        "photo of {trigger}, a woman smiling slightly, close-up with face visible, soft natural light from a window",
        "photo of {trigger}, a woman sitting in a cozy cafe, medium close-up from the chest up, wearing a casual knit sweater, warm afternoon light",
        "photo of {trigger}, a woman standing on a city street at dusk, medium shot from the waist up, wearing a dark wool coat, blurred city lights behind her",
        "photo of {trigger}, a woman laughing outdoors in a sunlit park, close-up with face visible, golden hour light, shallow depth of field",
    ],
    "man": [
        "photo of {trigger}, a man smiling slightly, close-up with face visible, soft natural light from a window",
        "photo of {trigger}, a man sitting in a cozy cafe, medium close-up from the chest up, wearing a casual button-up shirt, warm afternoon light",
        "photo of {trigger}, a man standing on a city street at dusk, medium shot from the waist up, wearing a dark wool coat, blurred city lights behind him",
        "photo of {trigger}, a man laughing outdoors in a sunlit park, close-up with face visible, golden hour light, shallow depth of field",
    ],
    "default": [
        "photo of {trigger}, a {class} smiling slightly, close-up with face visible, soft natural light from a window",
        "photo of {trigger}, a {class} standing on a city street at dusk, medium shot from the waist up, blurred city lights behind",
    ],
}


class Gallery:
    def __init__(self, config, people: Dict[str, str], output_dir: Path):
        """``people``: trigger -> class word (a finetune's prior.classes)."""
        config = config if isinstance(config, dict) else ({} if config in (None, True) else {"enabled": False})
        self.people = dict(sorted(people.items()))
        self.enabled = bool(config.get("enabled", True)) and bool(self.people)
        self.count = max(int(config.get("count", 8) or 8), 1)
        self.templates = {**DEFAULT_PROMPTS, **(config.get("prompts") or {})}
        self.dir = Path(output_dir) / "samples" / "gallery"
        self._random = random.Random()  # its own generator: a press never moves the training RNG

    def prompts_for(self, trigger: str) -> List[str]:
        cls = self.people[trigger]
        templates = self.templates.get(cls) or self.templates["default"]
        return [t.replace("{trigger}", trigger).replace("{class}", cls) for t in templates]

    @property
    def captions(self) -> List[str]:
        """Every prompt a press may use, to encode at the start."""
        if not self.enabled:
            return []
        return [p for trigger in self.people for p in self.prompts_for(trigger)]

    def run(
        self,
        step: int,
        render_one: Callable[[str, int], Image.Image],
        weights: Callable[[], ContextManager],
        ready: Callable[[str], bool] = lambda _prompt: True,
        label: str = "",
    ) -> float:
        """Render ``count`` random people with ``weights`` in place; returns the seconds it took."""
        started = time.time()
        picks = self._random.sample(list(self.people), min(self.count, len(self.people)))
        jobs = []
        for trigger in picks:
            options = [p for p in self.prompts_for(trigger) if ready(p)]
            if options:
                jobs.append((trigger, self._random.choice(options), self._random.randrange(2**31)))
        if not jobs:
            logger.warning("[gallery] nothing to render: the gallery prompts are not encoded")
            return 0.0
        self.dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[gallery] step %d: %d people%s: %s",
            step,
            len(jobs),
            f" ({label})" if label else "",
            ", ".join(t for t, _, _ in jobs),
        )
        images, lines = [], []
        with weights():
            for k, (trigger, prompt, seed) in enumerate(jobs, 1):
                one = time.time()
                image = render_one(prompt, seed, f"gallery {k}/{len(jobs)}")
                image.save(self.dir / f"step{step:06d}_{trigger}.png")
                images.append(image)
                lines.append(f"{trigger}  seed {seed}  {prompt}\n")
                logger.info("[gallery] %d/%d %s: %.0fs", k, len(jobs), trigger, time.time() - one)
        cols = min(4, len(images))
        rows = math.ceil(len(images) / cols)
        w, h = max(i.width for i in images), max(i.height for i in images)
        grid = Image.new("RGB", (cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, ((i % cols) * w, (i // cols) * h))
        grid.save(self.dir / f"step{step:06d}.png")
        (self.dir / f"step{step:06d}.txt").write_text("".join(lines), encoding="utf-8")
        elapsed = time.time() - started
        logger.info(
            "[gallery] step %d: %d images in %.0fs -> %s", step, len(images), elapsed, self.dir / f"step{step:06d}.png"
        )
        return elapsed
