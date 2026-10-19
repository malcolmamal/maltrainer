from __future__ import annotations

import asyncio
import warnings
from typing import Any

from starlette.exceptions import StarletteDeprecationWarning

# starlette 1.9 prefers httpx2, which is not installed here (and cannot be fetched offline); the
# deprecation fires once, at import, so it can only be filtered around the import itself.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="Using `httpx` with `starlette.testclient`", category=StarletteDeprecationWarning
    )
    from starlette.testclient import TestClient

_JOB_TASKS = {"TrainingJob.monitor", "TrainingJob.start_readers"}


class FakeStream:
    def __init__(self) -> None:
        self.chunks: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def read(self, size: int) -> bytes:
        chunk = await self.chunks.get() or b""
        assert len(chunk) <= size, "a fed chunk is larger than the size the app asked for"
        return chunk


class FakeProcess:
    def __init__(self, stdout: list[bytes], stderr: list[bytes], returncode: int, hold: bool) -> None:
        self.stdout, self.stderr = FakeStream(), FakeStream()
        self.returncode: int | None = None
        self.exit_code = returncode
        self.exited = asyncio.Event()
        for stream, chunks in ((self.stdout, stdout), (self.stderr, stderr)):
            for chunk in chunks:
                stream.chunks.put_nowait(chunk)
        if not hold:
            self.finish()

    def feed(self, chunk: bytes, stream: str = "stdout") -> None:
        getattr(self, stream).chunks.put_nowait(chunk)

    def finish(self, exit_code: int | None = None) -> None:
        if self.exited.is_set():
            return
        self.stdout.chunks.put_nowait(None)
        self.stderr.chunks.put_nowait(None)
        self.returncode = self.exit_code if exit_code is None else exit_code
        self.exited.set()

    async def wait(self) -> int | None:
        await self.exited.wait()
        return self.returncode


class FakeSpawner:
    """Stands in for asyncio.create_subprocess_exec: nothing a test does can start a real process."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.processes: list[FakeProcess] = []
        self.stdout: list[bytes] = []
        self.stderr: list[bytes] = []
        self.returncode = 0
        self.hold = False

    async def __call__(self, *args: str, **kwargs: Any) -> FakeProcess:
        self.calls.append((args, kwargs))
        process = FakeProcess(self.stdout, self.stderr, self.returncode, self.hold)
        self.processes.append(process)
        return process


def call_in_loop(client: TestClient, function: Any) -> Any:
    """Run a function on the event loop the client's app runs on (a client used as a context manager)."""
    assert client.portal is not None
    return client.portal.call(function)


def settle(client: TestClient) -> None:
    """Block until every job's monitor and log readers have run to the end."""

    async def _drain() -> None:
        current = asyncio.current_task()
        while pending := _job_tasks(current):
            await asyncio.wait_for(asyncio.gather(*pending), timeout=5)

    call_in_loop(client, _drain)


def _job_tasks(current: asyncio.Task | None) -> list[asyncio.Task]:
    return [
        t for t in asyncio.all_tasks() if t is not current and getattr(t.get_coro(), "__qualname__", "") in _JOB_TASKS
    ]
