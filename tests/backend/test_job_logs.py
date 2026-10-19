from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from backend import app as backend

from .conftest import add_job
from .fakes import FakeProcess


def _log(line: str, stream: str = "stdout", replace: bool = False, progress: bool = False) -> dict[str, Any]:
    return {"type": "log", "stream": stream, "line": line, "replace": replace, "is_progress": progress}


def _read(stdout: Sequence[bytes] = (), stderr: Sequence[bytes] = ()) -> backend.TrainingJob:
    job = add_job("j1", "running", process=FakeProcess(list(stdout), list(stderr), 0, hold=False))
    asyncio.run(job.start_readers())
    return job


def test_complete_lines_become_stdout_log_entries() -> None:
    job = _read([b"loading model\nready\n"])

    assert list(job.log_buffer) == [_log("loading model"), _log("ready")]


def test_a_line_split_across_chunks_is_reassembled() -> None:
    job = _read([b"loading mo", b"del\n"])

    assert list(job.log_buffer) == [_log("loading model")]


def test_stderr_lines_are_tagged_with_their_stream() -> None:
    job = _read(stderr=[b"warning: low memory\n"])

    assert list(job.log_buffer) == [_log("warning: low memory", stream="stderr")]


def test_a_carriage_return_marks_a_line_as_replacing_the_previous_one() -> None:
    job = _read([b"loading\rloaded\n"])

    assert list(job.log_buffer) == [_log("loading", replace=True), _log("loaded")]


def test_a_windows_line_ending_is_one_replacing_line_and_blank_lines_are_dropped() -> None:
    job = _read([b"first\r\n\n\nsecond\n"])

    assert list(job.log_buffer) == [_log("first", replace=True), _log("second")]


def test_a_final_line_without_a_newline_is_flushed_at_the_end_of_the_stream() -> None:
    job = _read([b"done\nbye"])

    assert list(job.log_buffer) == [_log("done"), _log("bye")]


def test_undecodable_bytes_are_dropped_from_a_line() -> None:
    job = _read([b"ok\xff\xfe!\n"])

    assert list(job.log_buffer) == [_log("ok!")]


def test_a_progress_line_updates_the_jobs_progress_and_is_broadcast_after_its_log_line() -> None:
    job = _read([b"Step: 5/20 loss=0.1\r"])

    assert list(job.log_buffer) == [
        _log("Step: 5/20 loss=0.1", replace=True, progress=True),
        {"type": "progress", "step": 5, "total": 20},
    ]
    assert job.progress == {"step": 5, "total": 20}


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (b"Step: 7/30\n", (7, 30)),
        (b"12 / 40 [00:01<00:02]\n", (12, 40)),
        (b"Epoch 3/10 done\n", (3, 10)),
        (b"no counters here\n", (0, 0)),
    ],
)
def test_progress_is_read_from_step_counters_in_the_output(line: bytes, expected: tuple[int, int]) -> None:
    job = _read([line])

    assert (job.progress["step"], job.progress["total"]) == expected


def test_the_last_progress_line_wins() -> None:
    job = _read([b"Step: 1/9\nStep: 2/9\n"], [b"Step: 3/9\n"])

    assert job.progress["step"] in (2, 3)
    assert job.progress["total"] == 9


def test_a_job_without_a_process_or_streams_reads_nothing() -> None:
    no_process = add_job("a", "running")
    no_streams = add_job("b", "running", process=FakeProcess([], [], 0, hold=False))
    no_streams.process.stdout = None  # type: ignore[union-attr]

    asyncio.run(no_process.start_readers())
    asyncio.run(no_streams.start_readers())

    assert (list(no_process.log_buffer), list(no_streams.log_buffer)) == ([], [])


def test_monitoring_a_job_without_a_process_changes_nothing() -> None:
    job = add_job("a", "queued")

    asyncio.run(job.monitor())

    assert (job.status, job.exit_code, job.ended_at) == ("queued", None, None)


def test_the_log_buffer_keeps_only_the_latest_two_thousand_entries() -> None:
    job = add_job("a", "running")

    for index in range(2005):
        asyncio.run(job.broadcast({"type": "log", "line": str(index)}))

    assert (len(job.log_buffer), job.log_buffer[0]["line"], job.log_buffer[-1]["line"]) == (2000, "5", "2004")


def test_a_broken_listener_is_dropped_and_the_others_keep_receiving() -> None:
    class Listener:
        def __init__(self, fail: bool) -> None:
            self.fail, self.got = fail, []

        async def send_json(self, payload: dict) -> None:
            if self.fail:
                raise RuntimeError("socket closed")
            self.got.append(payload)

    broken, healthy = Listener(True), Listener(False)
    job = add_job("a", "running", websockets={broken, healthy})

    asyncio.run(job.broadcast({"type": "status"}))

    assert (job.websockets, healthy.got, list(job.log_buffer)) == (
        {healthy},
        [{"type": "status"}],
        [{"type": "status"}],
    )
