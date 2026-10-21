"""The parent side of the mutation runner: worker processes and the pool that feeds them."""

from __future__ import annotations

import multiprocessing
import queue
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from types import TracebackType
from typing import Any

from maltrainer_quality.mutation_worker import READY, serve

Job = tuple[str, str, int | None, tuple[str, ...]]
START_TIMEOUT = 600.0


@dataclass(frozen=True)
class Outcome:
    """What running one job came to: pytest's exit code, how long, its output tail, or a timeout."""

    code: int
    seconds: float
    output: str
    timed_out: bool = False

    @property
    def survived(self) -> bool:
        return self.code == 0 and not self.timed_out


class Worker:
    def __init__(self, preload: Sequence[str]) -> None:
        self.preload = tuple(preload)
        self.process: BaseProcess
        self.connection: Any  # a Connection, or a PipeConnection on Windows
        self.start()

    def start(self) -> None:
        context = multiprocessing.get_context("spawn")
        mine, theirs = context.Pipe()
        self.process = context.Process(target=serve, args=(theirs, self.preload), daemon=True)
        self.process.start()
        self.connection = mine
        if not mine.poll(START_TIMEOUT) or mine.recv() != READY:
            self.stop()
            raise RuntimeError("a mutation worker did not start")

    def stop(self) -> None:
        self.process.kill()
        self.process.join()
        self.connection.close()

    def run(self, job: Job, timeout: float) -> Outcome:
        """A hung or crashed worker is replaced; a mutant that does that counts as killed."""
        try:
            self.connection.send(job)
            if self.connection.poll(timeout):
                code, seconds, output = self.connection.recv()
                return Outcome(code, seconds, output)
            reason = Outcome(1, timeout, "timed out", timed_out=True)
        except (EOFError, OSError):
            reason = Outcome(1, 0.0, "the worker process died")
        self.stop()
        self.start()
        return reason


class Pool:
    """``size`` workers; ``map`` runs jobs on whichever is free."""

    def __init__(self, size: int, preload: Sequence[str] = ("torch",)) -> None:
        self.free: queue.Queue[Worker] = queue.Queue()
        with ThreadPoolExecutor(size) as threads:  # each one imports torch: start them together
            self.workers = list(threads.map(Worker, [preload] * size))
        for worker in self.workers:
            self.free.put(worker)

    def __enter__(self) -> Pool:
        return self

    def __exit__(
        self, kind: type[BaseException] | None, error: BaseException | None, trace: TracebackType | None
    ) -> None:
        for worker in self.workers:
            worker.stop()

    def one(self, job: Job, timeout: float) -> Outcome:
        worker = self.free.get()
        try:
            return worker.run(job, timeout)
        finally:
            self.free.put(worker)

    def map(self, jobs: Sequence[Job], timeout: float) -> list[Outcome]:
        with ThreadPoolExecutor(len(self.workers)) as threads:
            return list(threads.map(lambda job: self.one(job, timeout), jobs))
