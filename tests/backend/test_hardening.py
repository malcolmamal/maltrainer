"""What the API refuses or survives: names that would leave the datasets folder or break the YAML
template, another web page's requests, a process that cannot start, and the order of a job's last
log lines and its exit status."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from backend import app as backend

from .conftest import add_job, write_images
from .fakes import FakeSpawner, TestClient, settle

BAD_NAMES = [
    "../outside_data",
    "..",
    ".hidden",
    "a/b",
    "a" + chr(92) + "b",
    'say "hi"',
    "line" + chr(10) + "break",
    "tab" + chr(13) + "name",
]
UI = "http://localhost:3003"


@pytest.mark.parametrize("name", BAD_NAMES)
@pytest.mark.parametrize("endpoint", ["/api/plan", "/api/train"])
def test_a_dataset_name_that_is_not_one_plain_folder_name_is_refused(
    client: TestClient, workspace: Path, name: str, endpoint: str
) -> None:
    write_images(workspace / "outside_data", 2)

    response = client.post(endpoint, json={"dataset": name, "dry_run": True})

    assert response.status_code == 422
    assert not (workspace / "output").exists()
    assert list((workspace / "outside_data").glob("*.yaml")) == []


@pytest.mark.parametrize("name", ["emma watson", "ben-2", "naïve", "o'brien", "a..b"])
def test_ordinary_folder_names_still_plan(client: TestClient, make_dataset: Callable[..., Path], name: str) -> None:
    make_dataset(name)

    response = client.post("/api/plan", json={"dataset": name})

    assert response.status_code == 200
    assert response.json()["dataset"] == name


def test_hidden_folders_in_the_datasets_root_are_not_datasets(
    client: TestClient, make_dataset: Callable[..., Path]
) -> None:
    for name in ("alice", ".git", ".trash"):
        make_dataset(name)

    assert [item["name"] for item in client.get("/api/datasets").json()] == ["alice"]


def test_the_ui_origin_may_call_the_api_and_another_page_may_not(client: TestClient) -> None:
    ask = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}

    ours = client.options("/api/train", headers={"Origin": UI, **ask})
    theirs = client.options("/api/train", headers={"Origin": "http://evil.example", **ask})

    assert ours.headers["access-control-allow-origin"] == UI
    assert "access-control-allow-credentials" not in ours.headers
    assert "access-control-allow-origin" not in theirs.headers
    assert theirs.status_code == 400


def test_a_job_whose_process_cannot_start_fails_and_does_not_block_the_next(
    client: TestClient, spawner: FakeSpawner, make_dataset: Callable[..., Path]
) -> None:
    make_dataset()
    spawner.fail_first = 1

    failed = client.post("/api/train", json={"dataset": "alice", "dry_run": True}).json()["job_id"]
    status = client.get(f"/api/jobs/{failed}").json()
    spawner.hold = True
    started = client.post("/api/train", json={"dataset": "alice", "dry_run": True}).json()["job_id"]

    assert (status["status"], status["exit_code"]) == ("failed", None)
    assert client.get(f"/api/jobs/{started}").json()["status"] == "running"


def test_a_queued_job_runs_after_the_one_before_it_failed_to_start(client: TestClient, spawner: FakeSpawner) -> None:
    add_job("first")
    add_job("second")
    backend.queued_job_ids.extend(["first", "second"])
    spawner.fail_first = 1
    spawner.hold = True

    client.post("/api/jobs/start-next")

    assert (backend.jobs["first"].status, backend.jobs["second"].status) == ("failed", "running")
    assert len(spawner.calls) == 2


def test_a_jobs_last_log_lines_come_before_the_status_that_says_it_ended(
    client: TestClient, spawner: FakeSpawner, make_dataset: Callable[..., Path]
) -> None:
    make_dataset()
    spawner.stdout = [b"first line\n", b"the very last line\n"]

    job_id = client.post("/api/train", json={"dataset": "alice", "dry_run": True}).json()["job_id"]
    settle(client)

    kinds = [(m["type"], m.get("line", m.get("status"))) for m in backend.jobs[job_id].log_buffer]
    assert kinds[-3:] == [("log", "first line"), ("log", "the very last line"), ("status", "exited")]


@pytest.mark.parametrize(
    ("line", "found"),
    [
        ("Step: 5/100", (5, 100)),
        ("training 12 / 400 [00:10]", (12, 400)),
        ("Loading shards 2/4", (2, 4)),
        ("started on 12/10/2025 at noon", None),
        ("see ./logs/3/4/run.txt", None),
        ("version 1.2/3.4", None),
        ("no counter here", None),
    ],
)
def test_only_a_step_counter_counts_as_progress(line: str, found: tuple[int, int] | None) -> None:
    match = backend.PROGRESS_RE.search(line)

    assert (match and (int(match[1]), int(match[2]))) == found or (match is None and found is None)
