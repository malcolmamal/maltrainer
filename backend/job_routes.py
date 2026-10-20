"""The routes that follow and steer jobs: their status, the queue, and the live log websocket."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from backend import training
from backend.schemas import JobListResponse, JobStatusResponse

router = APIRouter()


@router.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    job = training.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return training.job_to_status(job)


@router.get("/api/jobs", response_model=JobListResponse)
async def job_list() -> JobListResponse:
    items = [training.job_to_status(job) for job in training.jobs.values()]
    items.sort(key=lambda j: j.started_at, reverse=True)
    return JobListResponse(items=items)


@router.post("/api/jobs/{job_id}/start")
async def start_specific_job(job_id: str) -> dict[str, str]:
    job = training.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    async with training.job_lock:
        running = any(j.status == "running" for j in training.jobs.values())
        if running:
            raise HTTPException(status_code=409, detail="Another job is running")
        if job.status != "queued":
            raise HTTPException(status_code=400, detail="Job is not queued")
        training.queued_job_ids.remove(job_id) if job_id in training.queued_job_ids else None
        await training.launch_job(job)
    return {"status": "ok"}


@router.post("/api/jobs/start-next")
async def start_next_job() -> dict[str, str]:
    await training.launch_next_job()
    return {"status": "ok"}


@router.websocket("/ws/jobs/{job_id}")
async def job_ws(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    job = training.jobs.get(job_id)
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
                    "model": job.model,
                    "optimizer": job.optimizer,
                    "duration": job.duration,
                },
            }
        )
        for payload in job.log_buffer:
            await ws.send_json(payload)

        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        job.websockets.discard(ws)
    except (RuntimeError, OSError, KeyError):  # an error on the socket (or a binary frame): forget it and close
        job.websockets.discard(ws)
        await ws.close()
