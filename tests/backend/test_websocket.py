from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Any

import pytest
from starlette.websockets import WebSocketDisconnect

from backend import app as backend

from .conftest import add_job
from .fakes import FakeSpawner, TestClient, call_in_loop, settle

Maker = Callable[..., Any]


def _start(client: TestClient, spawner: FakeSpawner) -> str:
    spawner.hold = True
    response = client.post("/api/train", json={"dataset": "alice", "model": "zimage"})
    return response.json()["job_id"]


def test_an_unknown_job_gets_an_error_message_and_is_disconnected(client: TestClient) -> None:
    with client.websocket_connect("/ws/jobs/ghost") as ws:
        assert ws.receive_json() == {"type": "error", "message": "Job not found"}

        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_a_new_listener_gets_the_job_and_then_the_buffered_log_in_order() -> None:
    logs = [{"type": "log", "line": "a"}, {"type": "progress", "step": 1, "total": 2}]
    add_job(
        "j1", "running", 4.0, progress={"step": 1, "total": 2}, model="krea2", optimizer="prodigy", duration="2 steps"
    )
    backend.jobs["j1"].log_buffer = deque(logs)

    with TestClient(backend.app).websocket_connect("/ws/jobs/j1") as ws:
        first = ws.receive_json()
        replayed = [ws.receive_json(), ws.receive_json()]

    assert first == {
        "type": "job",
        "job": {
            "job_id": "j1",
            "dataset": "alice",
            "output_path": "out",
            "config_path": "out/config.yaml",
            "status": "running",
            "started_at": 4.0,
            "progress": {"step": 1, "total": 2},
            "model": "krea2",
            "optimizer": "prodigy",
            "duration": "2 steps",
        },
    }
    assert replayed == logs


def test_a_listener_is_registered_while_connected_and_removed_on_disconnect(client: TestClient) -> None:
    job = add_job("j1", "running")

    with client.websocket_connect("/ws/jobs/j1") as ws:
        ws.receive_json()
        assert len(job.websockets) == 1

    assert job.websockets == set()


def test_live_output_progress_and_the_final_status_are_streamed_as_they_happen(
    client: TestClient, make_dataset: Maker, spawner: FakeSpawner
) -> None:
    make_dataset()
    job_id = _start(client, spawner)
    process = spawner.processes[0]

    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        opening = ws.receive_json()
        call_in_loop(client, lambda: process.feed(b"Step: 3/400\n"))
        call_in_loop(client, lambda: process.feed(b"oops\n", "stderr"))
        streamed = [ws.receive_json() for _ in range(3)]
        call_in_loop(client, process.finish)
        closing = ws.receive_json()

    assert (opening["job"]["status"], opening["job"]["progress"]) == ("running", {"step": 0, "total": 400})
    assert streamed[0] == {
        "type": "log",
        "stream": "stdout",
        "line": "Step: 3/400",
        "replace": False,
        "is_progress": True,
    }
    assert streamed[1] == {"type": "progress", "step": 3, "total": 400}
    assert streamed[2]["stream"] == "stderr"
    assert closing == {"type": "status", "status": "exited", "exit_code": 0}


def test_every_connected_listener_receives_the_same_output(
    client: TestClient, make_dataset: Maker, spawner: FakeSpawner
) -> None:
    make_dataset()
    job_id = _start(client, spawner)
    with client.websocket_connect(f"/ws/jobs/{job_id}") as one, client.websocket_connect(f"/ws/jobs/{job_id}") as two:
        one.receive_json(), two.receive_json()

        call_in_loop(client, lambda: spawner.processes[0].feed(b"hello\n"))

        assert one.receive_json() == two.receive_json()


def test_a_listener_joining_after_the_run_ended_replays_all_of_it(
    client: TestClient, make_dataset: Maker, spawner: FakeSpawner
) -> None:
    make_dataset()
    spawner.stdout = [b"Step: 1/400\n"]
    spawner.returncode = 1
    job_id = client.post("/api/train", json={"dataset": "alice", "model": "zimage"}).json()["job_id"]
    settle(client)

    with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
        messages = [ws.receive_json() for _ in range(4)]

    assert messages[0]["job"]["status"] == "failed"
    assert sorted(message["type"] for message in messages[1:]) == ["log", "progress", "status"]
    assert {"type": "status", "status": "failed", "exit_code": 1} in messages


def test_a_binary_message_from_the_client_ends_the_session_and_unregisters_it(client: TestClient) -> None:
    job = add_job("j1", "running")

    with client.websocket_connect("/ws/jobs/j1") as ws:
        ws.receive_json()
        ws.send_bytes(b"\x00")

        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()

    assert job.websockets == set()
