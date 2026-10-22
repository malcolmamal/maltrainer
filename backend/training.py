"""Training jobs: a trainer process per job, its output read into log lines and progress for the websocket
clients, and the queue that starts the next job when one ends."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from backend.schemas import JobStatusResponse, StartTrainRequest, TrainPlan
from backend.specs import ModelSpec
from backend.streams import read_streams

logger = logging.getLogger("backend")

jobs: dict[str, TrainingJob] = {}
queued_job_ids: deque[str] = deque()
job_lock = asyncio.Lock()


@dataclass
class TrainingJob:
    job_id: str
    dataset: str
    output_path: str
    config_path: str
    command: list[str]
    process: asyncio.subprocess.Process | None
    started_at: float
    dry_run: bool
    dry_run_step_time: float
    model: str = "zimage"
    optimizer: str = ""
    duration: str = ""
    status: str = "running"
    ended_at: float | None = None
    exit_code: int | None = None
    progress: dict[str, Any] = field(default_factory=lambda: {"step": 0, "total": 0})
    websockets: set[WebSocket] = field(default_factory=set)
    log_buffer: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=2000))
    reader_task: asyncio.Task[None] | None = None
    monitor_task: asyncio.Task[None] | None = None

    async def start_readers(self) -> None:
        if self.process:
            await read_streams(self, self.process)

    async def monitor(self) -> None:
        if not self.process:
            return
        self.reader_task = asyncio.create_task(self.start_readers())
        await self.process.wait()
        # The last log lines come before the status that says the job is over.
        await asyncio.wait({self.reader_task}, timeout=5)
        self.exit_code = self.process.returncode
        self.ended_at = time.time()
        self.status = "exited" if self.exit_code == 0 else "failed"
        logger.info("Job finished: %s (%s) status=%s", self.job_id, self.dataset, self.status)
        await self.broadcast({"type": "status", "status": self.status, "exit_code": self.exit_code})
        await launch_next_job()

    async def broadcast(self, payload: dict[str, Any]) -> None:
        self.log_buffer.append(payload)
        stale: list[WebSocket] = []
        for ws in self.websockets:
            try:
                await ws.send_json(payload)
            except (RuntimeError, OSError, WebSocketDisconnect):  # the socket is gone
                stale.append(ws)
        for ws in stale:
            self.websockets.discard(ws)


def job_to_status(job: TrainingJob) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        dataset=job.dataset,
        output_path=job.output_path,
        config_path=job.config_path,
        status=job.status,
        started_at=job.started_at,
        ended_at=job.ended_at,
        exit_code=job.exit_code,
        progress=job.progress,
        model=job.model,
        optimizer=job.optimizer,
        duration=job.duration,
    )


async def spawn_process(command: list[str]) -> asyncio.subprocess.Process:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return await asyncio.create_subprocess_exec(
        *command,
        cwd=str(Path.cwd()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )


async def launch_job(job: TrainingJob) -> None:
    job.status = "running"
    job.started_at = time.time()
    try:
        job.process = await spawn_process(job.command)
    except OSError:
        logger.exception("Job %s could not start", job.job_id)
        job.status, job.ended_at = "failed", time.time()
        await job.broadcast({"type": "status", "status": "failed", "exit_code": None})
        return
    logger.info("Job started: %s (%s)", job.job_id, job.dataset)
    job.monitor_task = asyncio.create_task(job.monitor())


async def launch_next_job() -> None:
    async with job_lock:
        running = any(j.status == "running" for j in jobs.values())
        if running:
            return
        while queued_job_ids:
            next_id = queued_job_ids.popleft()
            job = jobs.get(next_id)
            if job and job.status == "queued":
                await launch_job(job)
                if job.status == "running":
                    return


def new_job(request: StartTrainRequest, plan: TrainPlan, spec: ModelSpec, config_path: Path) -> TrainingJob:
    """The job for a planned run (not started, and not yet in ``jobs``): the command line, the progress target."""
    cmd = [sys.executable, "-u", str(Path(spec.script).resolve()), "--config", str(config_path)]
    if request.dry_run:
        cmd.append("--dry-run")
        if request.dry_run_step_time is not None:
            cmd.extend(["--dry-run-step-time", str(request.dry_run_step_time)])
    job = TrainingJob(
        job_id=uuid.uuid4().hex,
        dataset=request.dataset,
        output_path=str(config_path.parent),
        config_path=str(config_path),
        command=cmd,
        process=None,
        started_at=time.time(),
        dry_run=request.dry_run,
        dry_run_step_time=request.dry_run_step_time,
        model=spec.id,
        optimizer=plan.optimizer,
        duration=plan.duration + (" · dry run" if request.dry_run else ""),
    )
    job.progress.update({"step": 0, "total": plan.total_steps})
    # Not "running" until launched - otherwise the check that finds a running job finds this one itself and
    # every submission queues, even on an idle trainer.
    job.status = "queued"
    return job
