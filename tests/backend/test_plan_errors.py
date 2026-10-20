from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend import registry, training

from .fakes import FakeSpawner


def _post(client: TestClient, **body: Any):
    return client.post("/api/plan", json={"dataset": "alice", **body})


def test_an_unknown_model_is_a_bad_request_listing_the_available_ones(client: TestClient) -> None:
    response = _post(client, model="sdxl")

    assert response.status_code == 400
    assert response.json()["detail"] == f"Unknown model 'sdxl'. Available: {', '.join(registry.MODELS)}"


def test_an_unknown_dataset_is_not_found(client: TestClient) -> None:
    response = _post(client, dataset="ghost")

    assert (response.status_code, response.json()["detail"]) == (404, "Dataset not found")


def test_a_file_in_the_datasets_folder_is_not_a_dataset(client: TestClient, workspace: Path) -> None:
    (workspace / "datasets" / "alice").write_text("a file, not a folder")

    assert _post(client).status_code == 404


def test_an_optimizer_the_model_does_not_offer_is_a_bad_request(client: TestClient, make_dataset: Callable) -> None:
    make_dataset()

    response = _post(client, model="zimage", optimizer="automagic3")

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Z-Image (Base) does not support optimizer 'automagic3'. Use one of: adamw8bit, prodigy"
    )


def test_the_model_is_checked_before_the_dataset(client: TestClient) -> None:
    assert _post(client, model="sdxl", dataset="ghost").status_code == 400


def test_a_missing_template_file_fails_the_plan_with_a_server_error(
    client: TestClient, make_dataset: Callable, tweak_spec: Callable
) -> None:
    make_dataset()
    tweak_spec("zimage", template="templates/nope.yaml")

    response = _post(client, model="zimage")

    assert (response.status_code, response.json()["detail"]) == (500, "Template not found: templates/nope.yaml")


@pytest.mark.parametrize(
    "body",
    [
        {"dataset": ""},
        {"learning_rate": 0},
        {"duration_mode": "weeks"},
        {"epochs": 0},
        {"epochs": 100001},
        {"max_steps": 0},
        {"save_every": 0},
    ],
)
def test_an_out_of_range_field_is_rejected_before_any_planning(client: TestClient, body: dict[str, Any]) -> None:
    response = client.post("/api/plan", json={"dataset": "alice", **body})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][0] == "body"


def test_a_plan_without_a_dataset_field_is_rejected(client: TestClient) -> None:
    assert client.post("/api/plan", json={}).status_code == 422


def test_a_failed_plan_starts_nothing_and_writes_nothing(
    client: TestClient, spawner: FakeSpawner, workspace: Path
) -> None:
    _post(client, dataset="ghost")

    assert (spawner.calls, training.jobs, (workspace / "output").exists()) == ([], {}, False)


def test_a_plan_previews_the_run_without_starting_or_writing_anything(
    client: TestClient, make_dataset: Callable, spawner: FakeSpawner, workspace: Path
) -> None:
    make_dataset(count=4)

    response = _post(client, model="zimage", epochs=10)

    assert response.status_code == 200
    assert response.json() == {
        "model": "zimage",
        "dataset": "alice",
        "optimizer": "adamw8bit",
        "learning_rate": 1e-4,
        "duration_mode": "epochs",
        "image_count": 4,
        "steps_per_epoch": 4,
        "epochs": 10,
        "total_steps": 40,
        "save_every": 500,
        "output_path": str(workspace / "output" / "alice"),
        "duration": "10 epochs (40 steps)",
        "auto_steps": False,
        "estimated_minutes": None,
        "notes": [],
        "warnings": [],
    }
    assert (spawner.calls, training.jobs, (workspace / "output").exists()) == ([], {}, False)


def test_a_null_dry_run_step_time_is_rejected(client: TestClient) -> None:
    assert _post(client, dry_run_step_time=None).status_code == 422
