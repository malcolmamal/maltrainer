"""One module under mutation testing: its path, the tests it is run against and its score floor."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    path: str
    tests: tuple[str, ...]
    floor: float
