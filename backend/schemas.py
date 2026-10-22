"""The request and response bodies of the web UI's API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    name: str
    path: str
    image_count: int


class OptimizerInfo(BaseModel):
    name: str
    label: str
    default_lr: float
    note: str


class ModelInfo(BaseModel):
    id: str
    label: str
    optimizers: list[OptimizerInfo]
    default_optimizer: str
    default_duration_mode: str
    default_epochs: int
    default_max_steps: int
    native_max_steps: bool
    sec_per_step: float
    models_root: str | None
    missing_files: list[str]
    auto_steps: bool
    target_epochs: int
    min_auto_steps: int
    max_auto_steps: int
    resolution_note: str


class StartTrainRequest(BaseModel):
    # One folder name inside the datasets folder: no separators, no leading dot ("..", ".git"), nothing that
    # breaks the YAML template the name is written into (quote, backslash, line break).
    dataset: str = Field(..., pattern=r'^[^./\\"\r\n\x00][^/\\"\r\n\x00]*$')
    model: str = "zimage"
    optimizer: str | None = None
    learning_rate: float | None = Field(None, gt=0)
    duration_mode: str | None = Field(None, pattern="^(epochs|steps)$")
    epochs: int | None = Field(None, ge=1, le=100000)
    max_steps: int | None = Field(None, ge=1, le=10000000)
    save_every: int | None = Field(None, ge=1)
    trigger: str | None = None
    dry_run: bool = False
    dry_run_step_time: float = 0.025


class TrainPlan(BaseModel):
    model: str
    dataset: str
    optimizer: str
    learning_rate: float
    duration_mode: str
    image_count: int
    steps_per_epoch: int
    epochs: int
    total_steps: int
    save_every: int
    output_path: str
    duration: str
    auto_steps: bool
    estimated_minutes: float | None
    notes: list[str]
    warnings: list[str]


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
    ended_at: float | None
    exit_code: int | None
    progress: dict[str, Any]
    model: str = "zimage"
    optimizer: str = ""
    duration: str = ""


class JobListResponse(BaseModel):
    items: list[JobStatusResponse]
