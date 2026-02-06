from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Set

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


DATASETS_ENV = "DATASETS_PATH"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PROGRESS_RE = re.compile(r"(?:Step:\s*)?(\d+)\s*/\s*(\d+)")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backend")


class DatasetInfo(BaseModel):
    name: str
    path: str
    image_count: int


class StartTrainRequest(BaseModel):
    dataset: str = Field(..., min_length=1)
    dry_run: bool = False
    dry_run_step_time: float = 0.025


class StartTrainResponse(BaseModel):
    job_id: str
    dataset: str
    output_path: str
    config_path: str
    command: list[str]


class JobStatusResponse(BaseModel):
    job_id: str
    dataset: str
    output_path: str
    config_path: str
    status: str
    started_at: float
    ended_at: Optional[float]
    exit_code: Optional[int]
    progress: Dict[str, Any]


class JobListResponse(BaseModel):
    items: list[JobStatusResponse]


@dataclass
class TrainingJob:
    job_id: str
    dataset: str
    output_path: str
    config_path: str
    command: list[str]
    process: Optional[asyncio.subprocess.Process]
    started_at: float
    dry_run: bool
    dry_run_step_time: float
    status: str = "running"
    ended_at: Optional[float] = None
    exit_code: Optional[int] = None
    progress: Dict[str, Any] = field(default_factory=lambda: {"step": 0, "total": 0})
    websockets: Set[WebSocket] = field(default_factory=set)
    log_buffer: Deque[dict] = field(default_factory=lambda: deque(maxlen=2000))
    reader_task: Optional[asyncio.Task] = None

    async def start_readers(self) -> None:
        if not self.process:
            return
        async def _emit_log(line: str, stream_name: str, replace: bool) -> None:
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
            await self.broadcast(payload)
            if match:
                step = int(match.group(1))
                total = int(match.group(2))
                self.progress.update({"step": step, "total": total})
                await self.broadcast({"type": "progress", "step": step, "total": total})

        async def _read_stream(stream: asyncio.StreamReader, stream_name: str) -> None:
            buffer = ""
            while True:
                chunk = await stream.read(1024)
                if not chunk:
                    if buffer:
                        await _emit_log(buffer, stream_name, replace=False)
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
                    await _emit_log(line, stream_name, replace=sep_char == "\r")

        stdout = self.process.stdout
        stderr = self.process.stderr
        if stdout is None or stderr is None:
            return
        await asyncio.gather(_read_stream(stdout, "stdout"), _read_stream(stderr, "stderr"))

    async def monitor(self) -> None:
        if not self.process:
            return
        self.reader_task = asyncio.create_task(self.start_readers())
        await self.process.wait()
        self.exit_code = self.process.returncode
        self.ended_at = time.time()
        self.status = "exited" if self.exit_code == 0 else "failed"
        logger.info("Job finished: %s (%s) status=%s", self.job_id, self.dataset, self.status)
        await self.broadcast(
            {
                "type": "status",
                "status": self.status,
                "exit_code": self.exit_code,
            }
        )
        await launch_next_job()

    async def broadcast(self, payload: dict) -> None:
        self.log_buffer.append(payload)
        stale: list[WebSocket] = []
        for ws in self.websockets:
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.websockets.discard(ws)


app = FastAPI(title="Maltrainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, TrainingJob] = {}
queued_job_ids: Deque[str] = deque()
job_lock = asyncio.Lock()


def datasets_root() -> Path:
    root = os.environ.get(DATASETS_ENV, "datasets")
    return Path(root).resolve()


def output_root() -> Path:
    return Path("output").resolve()


def load_template(dataset_name: str) -> Dict[str, Any]:
    template_path = Path("config") / "zimage_base_template.yaml"
    if not template_path.exists():
        raise FileNotFoundError("Template not found")
    template_text = template_path.read_text(encoding="utf-8")
    template_text = template_text.replace("[person_to_train]", dataset_name)
    config = yaml.safe_load(template_text)
    return config


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for root, _, files in os.walk(path):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTS:
                count += 1
    return count


def compute_total_steps(config: Dict[str, Any], image_count: int) -> int:
    dataset_cfg = config.get("dataset", {})
    train_cfg = config.get("train", {})
    repeats = int(dataset_cfg.get("repeats", 1) or 1)
    batch_size = int(train_cfg.get("batch_size", 1) or 1)
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1) or 1)
    num_epochs = int(train_cfg.get("num_epochs", 1) or 1)
    samples = max(image_count, 1) * max(repeats, 1)
    steps_per_epoch = math.ceil((samples / max(batch_size, 1)) / max(grad_accum, 1))
    return max(steps_per_epoch * num_epochs, 1)


async def spawn_process(command: list[str]) -> asyncio.subprocess.Process:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return await asyncio.create_subprocess_exec(
        *command,
        cwd=str(Path(".").resolve()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )


async def launch_job(job: TrainingJob) -> None:
    job.status = "running"
    job.started_at = time.time()
    job.process = await spawn_process(job.command)
    logger.info("Job started: %s (%s)", job.job_id, job.dataset)
    asyncio.create_task(job.monitor())


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
                return


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets", response_model=list[DatasetInfo])
async def list_datasets() -> list[DatasetInfo]:
    root = datasets_root()
    if not root.exists():
        return []
    items: list[DatasetInfo] = []
    for entry in root.iterdir():
        if entry.is_dir():
            items.append(
                DatasetInfo(
                    name=entry.name,
                    path=str(entry),
                    image_count=count_images(entry),
                )
            )
    return sorted(items, key=lambda d: d.name.lower())


@app.post("/api/train", response_model=StartTrainResponse)
async def start_training(request: StartTrainRequest) -> StartTrainResponse:
    root = datasets_root()
    dataset_path = root / request.dataset
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found")

    output_dir = output_root() / request.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_template(request.dataset)
    config.setdefault("dataset", {})
    config["dataset"]["path"] = str(dataset_path)
    config.setdefault("output", {})
    config["output"]["path"] = str(output_dir)

    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    image_count = count_images(dataset_path)
    total_steps = compute_total_steps(config, image_count)

    script_path = Path("train_zimage_lora.py").resolve()
    python_exe = sys.executable
    cmd = [python_exe, "-u", str(script_path), "--config", str(config_path)]
    if request.dry_run:
        cmd.append("--dry-run")
        if request.dry_run_step_time is not None:
            cmd.extend(["--dry-run-step-time", str(request.dry_run_step_time)])

    job_id = uuid.uuid4().hex
    job = TrainingJob(
        job_id=job_id,
        dataset=request.dataset,
        output_path=str(output_dir),
        config_path=str(config_path),
        command=cmd,
        process=None,
        started_at=time.time(),
        dry_run=request.dry_run,
        dry_run_step_time=request.dry_run_step_time,
    )
    job.progress.update({"step": 0, "total": total_steps})
    jobs[job_id] = job
    logger.info("Job added: %s (%s)", job.job_id, job.dataset)

    async with job_lock:
        running = any(j.status == "running" for j in jobs.values())
        if running:
            job.status = "queued"
            queued_job_ids.append(job_id)
            logger.info("Job queued: %s (%s)", job.job_id, job.dataset)
        else:
            await launch_job(job)

    return StartTrainResponse(
        job_id=job_id,
        dataset=request.dataset,
        output_path=str(output_dir),
        config_path=str(config_path),
        command=cmd,
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
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
    )


@app.get("/api/jobs", response_model=JobListResponse)
async def job_list() -> JobListResponse:
    items = [
        JobStatusResponse(
            job_id=job.job_id,
            dataset=job.dataset,
            output_path=job.output_path,
            config_path=job.config_path,
            status=job.status,
            started_at=job.started_at,
            ended_at=job.ended_at,
            exit_code=job.exit_code,
            progress=job.progress,
        )
        for job in jobs.values()
    ]
    items.sort(key=lambda j: j.started_at, reverse=True)
    return JobListResponse(items=items)


@app.post("/api/jobs/{job_id}/start")
async def start_specific_job(job_id: str) -> Dict[str, str]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    async with job_lock:
        running = any(j.status == "running" for j in jobs.values())
        if running:
            raise HTTPException(status_code=409, detail="Another job is running")
        if job.status != "queued":
            raise HTTPException(status_code=400, detail="Job is not queued")
        queued_job_ids.remove(job_id) if job_id in queued_job_ids else None
        await launch_job(job)
    return {"status": "ok"}


@app.post("/api/jobs/start-next")
async def start_next_job() -> Dict[str, str]:
    await launch_next_job()
    return {"status": "ok"}


@app.websocket("/ws/jobs/{job_id}")
async def job_ws(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    job = jobs.get(job_id)
    if not job:
        await ws.send_json({"type": "error", "message": "Job not found"})
        await ws.close()
        return

    job.websockets.add(ws)
    try:
        await ws.send_json(
            {
                "type": "job",
                "job": {
                    "job_id": job.job_id,
                    "dataset": job.dataset,
                    "output_path": job.output_path,
                    "config_path": job.config_path,
                    "status": job.status,
                    "started_at": job.started_at,
                    "progress": job.progress,
                },
            }
        )
        for payload in job.log_buffer:
            await ws.send_json(payload)

        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        job.websockets.discard(ws)
    except Exception:
        job.websockets.discard(ws)
        await ws.close()
