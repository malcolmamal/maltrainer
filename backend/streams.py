"""A trainer process's output as log lines and progress, from its two pipes: lines end at a newline or at a
carriage return (a progress bar redrawing itself replaces its last line)."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Protocol

# A step counter, not a date ("12/10/2025") or a path: no digit, "/" or "." touches either end.
PROGRESS_RE = re.compile(r"(?:Step:\s*)?(?<![\d/.])(\d+)\s*/\s*(\d+)(?![\d/.])")


class LogSink(Protocol):
    progress: dict[str, Any]

    async def broadcast(self, payload: dict[str, Any]) -> None: ...


async def read_streams(job: LogSink, process: asyncio.subprocess.Process) -> None:
    """Read stdout and stderr to their end, broadcasting every line through ``job``."""
    stdout, stderr = process.stdout, process.stderr
    if stdout is None or stderr is None:
        return
    await asyncio.gather(_read_stream(job, stdout, "stdout"), _read_stream(job, stderr, "stderr"))


async def _emit_log(job: LogSink, line: str, stream_name: str, replace: bool) -> None:
    text = line.rstrip("\n")
    if text == "":
        return
    match = PROGRESS_RE.search(text)
    is_progress = bool(match)
    payload = {
        "type": "log",
        "stream": stream_name,
        "line": text,
        "replace": replace,
        "is_progress": is_progress,
    }
    await job.broadcast(payload)
    if match:
        step = int(match.group(1))
        total = int(match.group(2))
        job.progress.update({"step": step, "total": total})
        await job.broadcast({"type": "progress", "step": step, "total": total})


async def _read_stream(job: LogSink, stream: asyncio.StreamReader, stream_name: str) -> None:
    buffer = ""
    while True:
        chunk = await stream.read(1024)
        if not chunk:
            if buffer:
                await _emit_log(job, buffer, stream_name, replace=False)
            break
        buffer += chunk.decode(errors="ignore")

        while True:
            nl = buffer.find("\n")
            cr = buffer.find("\r")
            if nl == -1 and cr == -1:
                break

            if nl == -1:
                sep_index = cr
                sep_char = "\r"
            elif cr == -1:
                sep_index = nl
                sep_char = "\n"
            else:
                sep_index = min(nl, cr)
                sep_char = "\r" if sep_index == cr else "\n"

            line = buffer[:sep_index]
            buffer = buffer[sep_index + 1 :]
            await _emit_log(job, line, stream_name, replace=sep_char == "\r")
