from __future__ import annotations

import asyncio
import dataclasses
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from backend import app as backend
from backend import files, registry, schemas, specs, training
from backend import plan as planner

from .fakes import FakeSpawner, TestClient, call_in_loop, settle

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_OUTPUT_ROOT = files.output_root


@pytest.fixture(autouse=True)
def spawner(monkeypatch: pytest.MonkeyPatch) -> FakeSpawner:
    fake = FakeSpawner()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake)
    return fake


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A datasets folder, a ComfyUI models folder and an output folder, all under tmp_path."""
    (tmp_path / "datasets").mkdir()
    (tmp_path / "models").mkdir()
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv(files.DATASETS_ENV, str(tmp_path / "datasets"))
    monkeypatch.setenv(files.COMFYUI_MODELS_ENV, str(tmp_path / "models"))
    monkeypatch.setattr(files, "output_root", lambda: tmp_path / "output")
    monkeypatch.setattr(training, "jobs", {})
    monkeypatch.setattr(training, "queued_job_ids", deque())
    monkeypatch.setattr(training, "job_lock", asyncio.Lock())
    monkeypatch.setattr(files, "_scan_cache", {})
    monkeypatch.setattr(files, "_model_file_cache", {})
    return tmp_path


@pytest.fixture
def client(spawner: FakeSpawner) -> Iterator[TestClient]:
    with TestClient(backend.app) as test_client:
        yield test_client
        spawner.hold = False
        for process in list(spawner.processes):
            call_in_loop(test_client, process.finish)
        settle(test_client)


def write_images(folder: Path, count: int, size: tuple[int, int] = (8, 8)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", size).save(folder / f"img{index}.png")
    return folder


@pytest.fixture
def make_dataset(workspace: Path) -> Callable[..., Path]:
    def make(name: str = "alice", count: int = 4, size: tuple[int, int] = (8, 8)) -> Path:
        return write_images(workspace / "datasets" / name, count, size)

    return make


@pytest.fixture
def install_models(workspace: Path) -> Callable[..., list[Path]]:
    """Create a model's checkpoint files under the models folder, optionally nested deeper."""

    def install(model_id: str, nested: str = "") -> list[Path]:
        paths = []
        for _, subfolder, name in registry.MODELS[model_id].model_files:
            path = workspace / "models" / subfolder / nested / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            paths.append(path)
        return paths

    return install


@pytest.fixture
def tweak_spec(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., specs.ModelSpec]:
    """Replace a model spec, and optionally merge sections into a copy of its template."""

    def tweak(model_id: str, sections: dict[str, dict[str, Any]] | None = None, **fields: Any) -> specs.ModelSpec:
        spec = registry.MODELS[model_id]
        if sections:
            config = yaml.safe_load((REPO_ROOT / spec.template).read_text(encoding="utf-8"))
            for section, values in sections.items():
                config.setdefault(section, {}).update(values)
            target = workspace / f"{model_id}_template.yaml"
            target.write_text(yaml.safe_dump(config), encoding="utf-8")
            fields["template"] = str(target)
        spec = dataclasses.replace(spec, **fields)
        monkeypatch.setitem(registry.MODELS, model_id, spec)
        return spec

    return tweak


def plan_of(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/api/plan", json={"dataset": "alice", **body})
    assert response.status_code == 200, response.text
    return response.json()


def build(**body: Any) -> tuple[schemas.TrainPlan, dict[str, Any]]:
    """Plan a run of the dataset "alice": the plan and the config the trainer would read."""
    plan, config, _, _ = planner.build_plan(schemas.StartTrainRequest(dataset="alice", **body))
    return plan, config


def add_job(job_id: str, status: str = "queued", started_at: float = 1.0, **fields: Any) -> training.TrainingJob:
    job = training.TrainingJob(
        job_id=job_id,
        dataset="alice",
        output_path="out",
        config_path="out/config.yaml",
        command=["python", "train.py"],
        process=fields.pop("process", None),
        started_at=started_at,
        dry_run=False,
        dry_run_step_time=0.025,
        status=status,
        **fields,
    )
    training.jobs[job_id] = job
    return job
