from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from backend import files, registry, training

from .conftest import REPO_ROOT
from .fakes import FakeSpawner

Maker = Callable[..., Any]


def _train(client: TestClient, **body: Any) -> Any:
    return client.post("/api/train", json={"dataset": "alice", "model": "zimage", **body})


def _config(response: Any) -> dict[str, Any]:
    return yaml.safe_load(Path(response.json()["config_path"]).read_text(encoding="utf-8"))


def test_training_writes_the_config_the_trainer_will_read(
    client: TestClient, make_dataset: Maker, workspace: Path
) -> None:
    make_dataset(count=4)

    response = _train(client, optimizer="prodigy", learning_rate=0.5, epochs=10, save_every=5, trigger="ohwx")

    config = _config(response)
    assert response.status_code == 200
    assert response.json()["config_path"] == str(workspace / "output" / "alice" / "config.yaml")
    assert response.json()["output_path"] == str(workspace / "output" / "alice")
    assert config["dataset"]["path"] == str(workspace / "datasets" / "alice")
    assert config["dataset"]["trigger"] == "ohwx"
    assert {key: config["train"][key] for key in ("optimizer", "learning_rate", "num_epochs", "save_every")} == {
        "optimizer": "prodigy",
        "learning_rate": 0.5,
        "num_epochs": 10,
        "save_every": 5,
    }
    assert config["output"]["path"] == response.json()["output_path"]


def test_training_runs_the_models_script_on_that_config_in_a_child_interpreter(
    client: TestClient, make_dataset: Maker, spawner: FakeSpawner
) -> None:
    make_dataset()

    response = _train(client)

    script = str((REPO_ROOT / "train_zimage_lora.py").resolve())
    expected = [sys.executable, "-u", script, "--config", response.json()["config_path"]]
    ((args, kwargs),) = spawner.calls
    assert (response.json()["command"], list(args)) == (expected, expected)
    assert kwargs["cwd"] == str(REPO_ROOT.resolve())
    assert (kwargs["stdout"], kwargs["stderr"]) == (subprocess.PIPE, subprocess.PIPE)
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"


@pytest.mark.parametrize("model_id", list(registry.MODELS))
def test_each_model_is_trained_by_its_own_script(client: TestClient, make_dataset: Maker, model_id: str) -> None:
    make_dataset()

    command = _train(client, model=model_id, dry_run=True).json()["command"]

    assert Path(command[2]).name == registry.MODELS[model_id].script


@pytest.mark.parametrize(("step_time", "expected"), [(None, "0.025"), (0.5, "0.5")])
def test_a_dry_run_passes_its_flags_and_is_labelled_in_the_job(
    client: TestClient, make_dataset: Maker, step_time: float | None, expected: str
) -> None:
    make_dataset()
    body = {} if step_time is None else {"dry_run_step_time": step_time}

    response = _train(client, dry_run=True, **body)

    assert response.json()["command"][-3:] == ["--dry-run", "--dry-run-step-time", expected]
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["duration"] == "100 epochs (400 steps) · dry run"


def test_a_real_run_gets_no_dry_run_flags(client: TestClient, make_dataset: Maker) -> None:
    make_dataset()

    assert "--dry-run" not in _train(client).json()["command"]


@pytest.mark.parametrize(("models_env", "shown"), [(True, "currently {root}"), (False, "currently unset")])
def test_training_without_its_checkpoints_is_refused_before_anything_is_written(
    client: TestClient,
    make_dataset: Maker,
    spawner: FakeSpawner,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    models_env: bool,
    shown: str,
) -> None:
    make_dataset()
    if not models_env:
        monkeypatch.delenv(files.COMFYUI_MODELS_ENV)

    response = _train(client, model="qwen_image_2_1")

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Cannot train Qwen-Image 2.1: missing model files diffusion_models/")
    assert shown.format(root=(workspace / "models").resolve()) in response.json()["detail"]
    assert (spawner.calls, training.jobs, (workspace / "output").exists()) == ([], {}, False)


def test_a_dry_run_does_not_need_the_checkpoints(client: TestClient, make_dataset: Maker) -> None:
    make_dataset()

    assert _train(client, model="qwen_image_2_1", dry_run=True).status_code == 200


def test_training_with_all_checkpoints_writes_their_paths_and_the_step_budget(
    client: TestClient, make_dataset: Maker, install_models: Maker
) -> None:
    make_dataset(count=10)
    paths = install_models("qwen_image_2_1")

    config = _config(_train(client, model="qwen_image_2_1", max_steps=600))

    assert config["model"]["transformer_path"] == str(paths[0])
    assert (config["train"]["max_steps"], config["train"]["num_epochs"]) == (600, 60)


def test_two_runs_of_a_per_run_model_get_separate_output_folders(client: TestClient, make_dataset: Maker) -> None:
    make_dataset()

    first = _train(client, model="krea2", dry_run=True).json()
    second = _train(client, model="krea2", dry_run=True).json()

    assert first["output_path"] != second["output_path"]
    assert second["output_path"].endswith("-2")
    assert Path(first["config_path"]).is_file() and Path(second["config_path"]).is_file()


@pytest.mark.parametrize(
    ("body", "status"), [({"dataset": "ghost"}, 404), ({"model": "sdxl"}, 400), ({"epochs": 0}, 422)]
)
def test_a_rejected_request_starts_and_writes_nothing(
    client: TestClient, make_dataset: Maker, spawner: FakeSpawner, workspace: Path, body: dict[str, Any], status: int
) -> None:
    make_dataset()

    response = _train(client, **body)

    assert response.status_code == status
    assert (spawner.calls, training.jobs, (workspace / "output").exists()) == ([], {}, False)
