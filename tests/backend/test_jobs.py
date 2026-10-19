from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from .conftest import add_job
from .fakes import FakeSpawner, settle

Maker = Callable[..., Any]


def _train(client: TestClient, dataset: str = "alice") -> dict[str, Any]:
    response = client.post("/api/train", json={"dataset": dataset, "model": "zimage"})
    assert response.status_code == 200
    return response.json()


def _statuses(client: TestClient) -> dict[str, str]:
    return {job["job_id"]: job["status"] for job in client.get("/api/jobs").json()["items"]}


def _configs(spawner: FakeSpawner) -> list[str]:
    return [args[-1] for args, _ in spawner.calls]


def test_there_are_no_jobs_before_anything_is_started(client: TestClient) -> None:
    assert client.get("/api/jobs").json() == {"items": []}


def test_jobs_are_listed_newest_first(client: TestClient) -> None:
    for job_id, started_at in (("old", 1.0), ("new", 3.0), ("mid", 2.0)):
        add_job(job_id, "exited", started_at)

    assert [job["job_id"] for job in client.get("/api/jobs").json()["items"]] == ["new", "mid", "old"]


def test_a_job_reports_its_status_progress_and_model(client: TestClient) -> None:
    add_job("j1", "failed", 5.0, exit_code=2, ended_at=9.0, optimizer="prodigy", duration="3 steps")

    body = client.get("/api/jobs/j1").json()

    assert body == {
        "job_id": "j1",
        "dataset": "alice",
        "output_path": "out",
        "config_path": "out/config.yaml",
        "status": "failed",
        "started_at": 5.0,
        "ended_at": 9.0,
        "exit_code": 2,
        "progress": {"step": 0, "total": 0},
        "model": "zimage",
        "optimizer": "prodigy",
        "duration": "3 steps",
    }


@pytest.mark.parametrize(("method", "path"), [("get", "/api/jobs/ghost"), ("post", "/api/jobs/ghost/start")])
def test_an_unknown_job_id_is_not_found(client: TestClient, method: str, path: str) -> None:
    response = getattr(client, method)(path)

    assert (response.status_code, response.json()) == (404, {"detail": "Job not found"})


def test_a_job_that_exits_cleanly_is_marked_exited_with_its_last_progress(
    client: TestClient, make_dataset: Maker, spawner: FakeSpawner
) -> None:
    make_dataset()
    spawner.stdout = [b"Step: 1/400\n", b"Step: 400/400\n"]

    job_id = _train(client)["job_id"]
    settle(client)

    job = client.get(f"/api/jobs/{job_id}").json()
    assert (job["status"], job["exit_code"], job["progress"]) == ("exited", 0, {"step": 400, "total": 400})
    assert job["ended_at"] >= job["started_at"]


def test_a_job_with_a_nonzero_exit_code_is_marked_failed(
    client: TestClient, make_dataset: Maker, spawner: FakeSpawner
) -> None:
    make_dataset()
    spawner.returncode = 3

    job_id = _train(client)["job_id"]
    settle(client)

    job = client.get(f"/api/jobs/{job_id}").json()
    assert (job["status"], job["exit_code"]) == ("failed", 3)
