"""
Per-folder repeats: how often each folder's images appear in one epoch.

    dataset:
      repeats: 1                       # the default for every folder
      folder_repeats:                  # by path under dataset.path, or by the folder's own name
        people/ben: 3                  # trained worst: three times per epoch
        emmawatson: 0.5                # trained well: half of her images each epoch

A repeat count can be fractional: 2.5 shows every image twice plus a random half of the folder
once more; 0.5 shows a random half. The half is drawn again every epoch, so all images are used
over a run. Uses Python's ``random`` (a resumed run restores it).
"""

from __future__ import annotations

import math
import random
from pathlib import PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple


def folder_repeats(rel_folder: str, table: Optional[Dict[str, float]], default: float) -> float:
    """The repeats for a folder (its posix path relative to the dataset root): an exact path entry
    first, then one naming the folder alone, else ``default``."""
    table = table or {}
    rel_folder = PurePosixPath(rel_folder).as_posix()
    if rel_folder in table:
        return float(table[rel_folder])
    name = PurePosixPath(rel_folder).name
    return float(table.get(name, default))


def unknown_keys(table: Optional[Dict[str, float]], rel_folders: Sequence[str]) -> List[str]:
    """``folder_repeats`` entries that match no folder (a typo would silently do nothing)."""
    known = set(rel_folders) | {PurePosixPath(f).name for f in rel_folders}
    return sorted(k for k in (table or {}) if PurePosixPath(k).as_posix() not in known)


def samples_per_epoch(count: int, repeats: float) -> int:
    whole = math.floor(repeats)
    return count * whole + round((repeats - whole) * count)


def repeat_order(groups: Sequence[Tuple[Sequence[int], float]]) -> List[int]:
    """One epoch's shuffled sample indices from (indices, repeats) groups."""
    order: List[int] = []
    for indices, repeats in groups:
        indices = list(indices)
        whole = math.floor(max(repeats, 0.0))
        order += indices * whole
        extra = round((max(repeats, 0.0) - whole) * len(indices))
        if extra:
            order += random.sample(indices, extra)
    random.shuffle(order)
    return order
