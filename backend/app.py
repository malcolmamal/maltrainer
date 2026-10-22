"""The web UI's API: datasets, models, a plan preview, starting training, and following jobs."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend import files, job_routes, training
from backend.plan import build_plan
from backend.registry import MODELS
from backend.schemas import (
    DatasetInfo,
    ModelInfo,
    OptimizerInfo,
    StartTrainRequest,
    StartTrainResponse,
    TrainPlan,
)
from backend.specs import ModelSpec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backend")

app = FastAPI(title="Maltrainer API")
app.include_router(job_routes.router)

app.add_middleware(
    CORSMiddleware,
    # The UI's own origin (it normally reaches us through its dev-server proxy, same origin): another
    # web page open in the browser must not be able to start training here.
    allow_origins=["http://localhost:3003", "http://127.0.0.1:3003"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets", response_model=list[DatasetInfo])
async def list_datasets() -> list[DatasetInfo]:
    root = files.datasets_root()
    if not root.exists():
        return []
    items: list[DatasetInfo] = []
    for entry in root.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            items.append(DatasetInfo(name=entry.name, path=str(entry), image_count=files.count_images(entry)))
    return sorted(items, key=lambda d: d.name.lower())


def model_info(spec: ModelSpec) -> ModelInfo:
    _, missing = files.resolve_model_files(spec) if spec.model_files else ({}, [])
    root = files.comfyui_models_root()
    return ModelInfo(
        id=spec.id,
        label=spec.label,
        optimizers=[
            OptimizerInfo(name=o.name, label=o.label, default_lr=o.default_lr, note=o.note) for o in spec.optimizers
        ],
        default_optimizer=spec.default_optimizer,
        default_duration_mode=spec.default_duration_mode,
        default_epochs=spec.default_epochs,
        default_max_steps=spec.default_max_steps,
        native_max_steps=spec.native_max_steps,
        sec_per_step=spec.sec_per_step,
        models_root=str(root) if (root and spec.model_files) else None,
        missing_files=missing,
        auto_steps=bool(spec.target_epochs),
        target_epochs=spec.target_epochs,
        min_auto_steps=spec.min_auto_steps,
        max_auto_steps=spec.max_auto_steps,
        resolution_note=spec.resolution_note,
    )


@app.get("/api/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    return [model_info(spec) for spec in MODELS.values()]


@app.post("/api/plan", response_model=TrainPlan)
async def plan_training(request: StartTrainRequest) -> TrainPlan:
    """Preview what /api/train would do with this request, without starting anything."""
    plan, _, _, _ = build_plan(request)
    return plan


@app.post("/api/train", response_model=StartTrainResponse)
async def start_training(request: StartTrainRequest) -> StartTrainResponse:
    plan, config, spec, missing = build_plan(request)
    if missing and not request.dry_run:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot train {spec.label}: missing model files {', '.join(missing)}. "
                f"Set {files.COMFYUI_MODELS_ENV} to your ComfyUI models folder "
                f"(currently {files.comfyui_models_root() or 'unset'}) and restart the backend."
            ),
        )

    output_dir = Path(plan.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    job = training.new_job(request, plan, spec, config_path)
    training.jobs[job.job_id] = job
    logger.info("Job added: %s (%s)", job.job_id, job.dataset)

    async with training.job_lock:
        running = any(j.status == "running" for j in training.jobs.values() if j is not job)
        if running:
            job.status = "queued"
            training.queued_job_ids.append(job.job_id)
            logger.info("Job queued: %s (%s)", job.job_id, job.dataset)
        else:
            await training.launch_job(job)

    return StartTrainResponse(
        job_id=job.job_id,
        dataset=request.dataset,
        output_path=str(output_dir),
        config_path=str(config_path),
        command=job.command,
    )
