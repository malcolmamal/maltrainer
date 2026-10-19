"""Walking a dataset folder the same way everywhere."""

from __future__ import annotations

import os
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Union


def walk_visible(root: Union[str, Path]) -> Iterator[tuple[str, list[str], list[str]]]:
    """``os.walk(root)`` without the hidden folders below it (names starting with "."), so a
    ``.ipynb_checkpoints`` full of copies, a ``.trash`` or the ``.maltrainer_cache`` is never
    mistaken for training images. ``root`` itself may be hidden or sit inside a hidden folder."""
    for folder, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if not name.startswith(".")]
        yield folder, dirs, files


def count_files(root: Union[str, Path], extensions: Collection[str]) -> int:
    """Files under ``root`` (hidden folders skipped) whose suffix, in any case, is in ``extensions``."""
    return sum(Path(name).suffix.lower() in extensions for _, _, files in walk_visible(root) for name in files)
