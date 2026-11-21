"""File-system helpers for writes that must not fail on a passing condition."""

from __future__ import annotations

import contextlib
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image

logger = logging.getLogger(__name__)


def save_or_report(image: Image.Image, path: Path) -> bool:
    """``image.save(path)``; a write the disk refuses (full, or the file locked) is logged and removed, never
    raised: a preview that cannot be written must not end a training run."""
    try:
        image.save(path)
        return True
    except OSError as error:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        logger.error("%s could not be written (%s); training continues", path, error)
        return False


def torch_save_closed(value: Any, path: Path) -> None:
    """``torch.save`` into a file this function closes, also when the write fails (given a path, torch can keep
    the file open while the error is handled, and Windows cannot delete an open file)."""
    with path.open("wb") as file:
        torch.save(value, file)


def remove_unfinished(folder: Path, what: str) -> None:
    """Delete a folder a failed write left; one that cannot be deleted is logged with its path."""
    shutil.rmtree(folder, ignore_errors=True)
    if folder.exists():
        logger.error("Could not remove %s, the unfinished %s: delete it to free its space.", folder, what)


def rename_with_retry(
    source: Path,
    target: Path,
    attempts: int = 5,
    delay: float = 1.0,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """``source.rename(target)``, tried again when the OS refuses with a permission error.

    On Windows a virus scanner or the search indexer can hold a freshly written folder for a moment
    (WinError 5 / 32); the same rename succeeds a second later. The waits grow (``delay``, 2 x ``delay``,
    ...), and the last failure is raised as it is.
    """
    for attempt in range(1, attempts + 1):
        try:
            source.rename(target)
            return
        except PermissionError as error:
            if attempt == attempts:
                raise
            logger.warning(
                "Renaming %s to %s was refused (%s); trying again in %.1f s", source, target, error, delay * attempt
            )
            (sleep or time.sleep)(delay * attempt)


def replace_directory(finished: Path, target: Path) -> None:
    """Put the finished folder in the place of ``target``, which stays whole until the new one is in.

    The old ``target`` is moved to ``<target>.old``, the finished folder takes its place, and the old
    one is deleted last, so a failure in between leaves a complete folder behind. Both renames are
    retried (see ``rename_with_retry``).
    """
    old = target.with_name(target.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        rename_with_retry(target, old)
    rename_with_retry(finished, target)
    shutil.rmtree(old, ignore_errors=True)
