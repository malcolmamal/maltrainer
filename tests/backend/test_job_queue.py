from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend import app as backend

from .conftest import add_job
from .fakes import FakeSpawner, call_in_loop, settle

Maker = Callable[..., Any]


def _train(client: TestClient, dataset: str = "alice") -> dict[str, Any]:
    response = client.post("/api/train", json={"dataset": dataset, "model": "zimage"})
    assert response.status_code == 200
    return response.json()


def _statuses(client: TestClient) -> dict[str, str]:
    return {job["job_id"]: job["status"] for job in client.get("/api/jobs").json()["items"]}


def _configs(spawner: FakeSpawner) -> list[str]:
    return [args[-1] for args, _ in spawner.calls]


def test_a_new_job_waits_for_the_running_one_and_then_runs_in_submission_order(
    client: TestClient, make_dataset: Maker, spawner: FakeSpawner
) -> None:
    ids = {}
    for name in ("alice", "bob", "carol"):
        make_dataset(name)
    spawner.hold = True
    ids["alice"] = _train(client, "alice")
    ids["bob"] = _train(client, "bob")
    ids["carol"] = _train(client, "carol")

    assert [_statuses(client)[ids[name]["job_id"]] for name in ids] == ["running", "queued", "queued"]
    assert len(spawner.calls) == 1
    assert list(backend.queued_job_ids) == [ids["bob"]["job_id"], ids["carol"]["job_id"]]

    spawner.hold = False
    call_in_loop(client, spawner.processes[0].finish)
    settle(client)

    assert set(_statuses(client).values()) == {"exited"}
    assert _configs(spawner) == [ids[name]["config_path"] for name in ids]
    assert not backend.queued_job_ids


def test_starting_a_queued_job_by_id_runs_it_and_takes_it_off_the_queue(
    client: TestClient, spawner: FakeSpawner
) -> None:
    spawner.hold = True
    add_job("q1")
    backend.queued_job_ids.append("q1")

    response = client.post("/api/jobs/q1/start")

    assert (response.status_code, response.json()) == (200, {"status": "ok"})
    assert [list(args) for args, _ in spawner.calls] == [["python", "train.py"]]
    assert (client.get("/api/jobs/q1").json()["status"], list(backend.queued_job_ids)) == ("running", [])


def test_a_queued_job_missing_from_the_queue_can_still_be_started(client: TestClient, spawner: FakeSpawner) -> None:
    spawner.hold = True
    add_job("q1")

    assert client.post("/api/jobs/q1/start").status_code == 200
    assert len(spawner.calls) == 1


def test_an_unqueued_job_cannot_be_started(client: TestClient, spawner: FakeSpawner) -> None:
    add_job("target", "exited")

    response = client.post("/api/jobs/target/start")

    assert (response.status_code, response.json()["detail"], spawner.calls) == (400, "Job is not queued", [])


def test_a_job_cannot_be_started_while_another_one_runs(client: TestClient, spawner: FakeSpawner) -> None:
    add_job("busy", "running")
    add_job("target")

    response = client.post("/api/jobs/target/start")

    assert (response.status_code, response.json()["detail"], spawner.calls) == (409, "Another job is running", [])


def test_start_next_runs_the_oldest_queued_job_and_skips_stale_queue_entries(
    client: TestClient, spawner: FakeSpawner
) -> None:
    spawner.hold = True
    add_job("done", "exited")
    add_job("q1")
    add_job("q2")
    backend.queued_job_ids.extend(["ghost", "done", "q1", "q2"])

    assert client.post("/api/jobs/start-next").json() == {"status": "ok"}

    assert (_statuses(client)["q1"], _statuses(client)["q2"], list(backend.queued_job_ids)) == (
        "running",
        "queued",
        ["q2"],
    )
    assert len(spawner.calls) == 1


@pytest.mark.parametrize("running", [True, False])
def test_start_next_does_nothing_while_a_job_runs_or_when_nothing_is_queued(
    client: TestClient, spawner: FakeSpawner, running: bool
) -> None:
    if running:
        add_job("busy", "running")
        add_job("q1")
        backend.queued_job_ids.append("q1")

    assert client.post("/api/jobs/start-next").json() == {"status": "ok"}
    assert (spawner.calls, list(backend.queued_job_ids)) == ([], ["q1"] if running else [])
