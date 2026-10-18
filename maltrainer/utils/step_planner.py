"""
--suggest-steps: count a dataset's images per folder and suggest ``train.max_steps`` for a target
number of epochs, without loading any model.

An epoch is every image of every group times its repeats (e.g. the subjects ``repeats`` times
and the regularisation images ``reg_repeats`` times), so max_steps = epochs x steps per epoch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from maltrainer.data.minimax_dataset import _media_kind, count_media_files

DEFAULT_EPOCHS = 22  # the MOSS-Image people LoRAs had likeness by ~20 epochs


@dataclass
class Group:
    label: str
    root: str
    repeats: int
    folders: List[Tuple[str, int]] = field(default_factory=list)  # (sub-folder, images)

    @property
    def images(self) -> int:
        return sum(n for _, n in self.folders)

    @property
    def samples(self) -> int:
        return self.images * self.repeats


def folder_counts(root: Union[str, Path], kinds: Sequence[str] = ("image",)) -> List[Tuple[str, int]]:
    """Images per immediate sub-folder of ``root`` (each counted with everything below it), and
    "." for the files directly in it. Hidden folders such as .maltrainer_cache are skipped."""
    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    top = sum(1 for p in root.iterdir() if p.is_file() and _media_kind(p) in set(kinds))
    if top:
        out.append((".", top))
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        n = count_media_files(sub, kinds)
        if n:
            out.append((sub.name, n))
    return out


def report(
    groups: Sequence[Group],
    train_config: dict,
    epochs: Optional[float] = None,
    notes: Sequence[str] = (),
) -> str:
    """The per-folder counts, the epoch size and the suggested max_steps, as printable text."""
    epochs = float(epochs or DEFAULT_EPOCHS)
    accum = max(int(train_config.get("gradient_accumulation_steps", 1) or 1), 1)
    lines = []
    for g in groups:
        lines.append(f"{g.label}: {g.root} ({g.images} images x {g.repeats} repeats = {g.samples} samples per epoch)")
        width = max((len(name) for name, _ in g.folders), default=0)
        for name, n in g.folders:
            lines.append(f"  {name:<{width}}  {n:>5}")
        if not g.folders:
            lines.append("  (no images found)")
    samples = sum(g.samples for g in groups)
    steps_per_epoch = max(math.ceil(samples / accum), 1)
    suggested = math.ceil(epochs * steps_per_epoch)
    lines.append("")
    lines.append(f"One epoch: {samples} samples = {steps_per_epoch} steps"
                 + (f" (gradient_accumulation_steps {accum})" if accum > 1 else ""))
    lines.append(f"Suggested for {epochs:g} epochs:  train.max_steps: {suggested}")
    current = int(train_config.get("max_steps", 0) or 0)
    if current > 0:
        lines.append(f"Configured now:  max_steps {current} = {current / steps_per_epoch:.1f} epochs")
    else:
        num_epochs = int(train_config.get("num_epochs", 10) or 1)
        lines.append(f"Configured now:  max_steps 0, so num_epochs {num_epochs} = {num_epochs * steps_per_epoch} steps")
    for note in notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines)
