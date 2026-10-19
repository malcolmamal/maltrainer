"""Stand-ins for the console that maltrainer.utils.keyboard reads: msvcrt, a TTY stdin and a clock."""

from __future__ import annotations

import io
from typing import List


class FakeMsvcrt:
    """Key presses arrive in batches: the first is waiting at once, each sleep delivers the next."""

    def __init__(self, *batches: str) -> None:
        self.batches = list(batches)
        self.pending: List[str] = []
        self.deliver()

    def deliver(self) -> None:
        if not self.batches:
            raise AssertionError("the code waited for a key that never comes")
        self.pending.extend(self.batches.pop(0))

    def kbhit(self) -> bool:
        return bool(self.pending)

    def getwch(self) -> str:
        return self.pending.pop(0)


class FakeClock:
    def __init__(self, console: FakeMsvcrt | None = None) -> None:
        self.now = 100.0
        self.console = console
        self.sleeps: List[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        if self.console is not None:
            self.console.deliver()


class FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return 7
