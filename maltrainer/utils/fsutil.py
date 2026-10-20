"""File-system helpers for writes that must not fail on a passing condition."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


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
