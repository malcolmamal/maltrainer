from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from maltrainer.control.hub import Capabilities, Command, CommandState, ControlHub, RunState
from maltrainer.control.previews_patch import preview_snapshot
from maltrainer.utils.previews import Previews

from .helpers import client_for, make_view


def _hub_with_previews(tmp_path: Path) -> ControlHub:
    previews = Previews({"every": 100, "prompts": ["a", {"prompt": "b", "enabled": False}]}, tmp_path)
    hub = ControlHub(Capabilities(gallery=True, preview_settings=True))
    hub.publish(settings=preview_snapshot(previews, ["raw", "ema"]))
    return hub


def test_status_before_the_first_step_has_the_state_and_what_the_trainer_can_do() -> None:
    body = client_for(ControlHub(Capabilities(gallery=True))).get("/status").json()

    assert body["state"] == "running"
    assert body["step"] is None
    assert body["published_seconds_ago"] is None
    assert body["capabilities"] == {"gallery": True, "preview_settings": False}


def test_status_reports_the_published_stats_and_the_live_state() -> None:
    hub = ControlHub()
    hub.publish(stats=make_view(total=200).stats(50))
    hub.state = RunState.PAUSED

    body = client_for(hub).get("/status").json()

    assert (body["state"], body["step"], body["total_steps"], body["percent"]) == ("paused", 50, 200, 25.0)
    assert body["epoch"] == 1
    assert body["learning_rate"] == pytest.approx(0.01)
    assert body["window"]["window_steps"] == 0
    assert body["published_seconds_ago"] is not None and body["published_seconds_ago"] >= 0


@pytest.mark.parametrize("action", ["save", "save-and-stop", "pause", "gallery"])
def test_each_command_is_accepted_with_202_and_queued(action: str) -> None:
    hub = ControlHub(Capabilities(gallery=True))

    response = client_for(hub).post(f"/commands/{action}")

    assert response.status_code == 202
    body = response.json()
    assert (body["command"], body["state"]) == (action, "pending")
    assert [p.record.id for p in hub.take(paused=False)] == [body["id"]]


def test_a_command_can_be_followed_until_it_is_done() -> None:
    hub = ControlHub()
    client = client_for(hub)
    command_id = client.post("/commands/save").json()["id"]
    assert client.get(f"/commands/{command_id}").json()["state"] == "pending"

    (pending,) = hub.take(paused=False)
    hub.settle(pending.record, True, "saved")

    body = client.get(f"/commands/{command_id}").json()
    assert (body["state"], body["detail"]) == ("done", "saved")
    assert body["settled_at"] is not None
    assert [c["id"] for c in client.get("/commands").json()] == [command_id]


def test_an_unknown_command_id_is_404_and_an_unknown_action_is_422() -> None:
    client = client_for(ControlHub())

    assert client.get("/commands/99").status_code == 404
    assert client.post("/commands/explode").status_code == 422


def test_a_command_the_run_cannot_take_is_409_with_the_reason() -> None:
    hub = ControlHub()
    client = client_for(hub)

    gallery = client.post("/commands/gallery")
    hub.state = RunState.STOPPING
    save = client.post("/commands/save")

    assert (gallery.status_code, save.status_code) == (409, 409)
    assert "no gallery" in gallery.json()["detail"]
    assert "stopping" in save.json()["detail"]


def test_preview_settings_are_listed_and_changes_are_queued_as_a_command(tmp_path: Path) -> None:
    hub = _hub_with_previews(tmp_path)
    client = client_for(hub)
    assert client.get("/preview-settings").json() == {
        "every": 100,
        "rows": {"raw": True, "ema": True},
        "prompts": [{"slot": 0, "prompt": "a", "enabled": True}, {"slot": 1, "prompt": "b", "enabled": False}],
    }

    response = client.patch(
        "/preview-settings",
        json={"every": 50, "rows": {"ema": False}, "prompts": [{"slot": 1, "enabled": True}]},
    )

    assert response.status_code == 202
    (pending,) = hub.take(paused=False)
    assert pending.record.command is Command.PREVIEW_SETTINGS
    assert pending.record.state is CommandState.PENDING
    assert pending.record.payload == {"every": 50, "rows": {"ema": False}, "prompts": [{"slot": 1, "enabled": True}]}


@pytest.mark.parametrize(
    "patch",
    [
        {},
        {"every": -1},
        {"prompts": [{"slot": 7, "enabled": True}]},
        {"rows": {"bogus": True}},
        {"unknown": 1},
    ],
)
def test_a_preview_settings_change_that_makes_no_sense_is_422_and_queues_nothing(
    tmp_path: Path, patch: dict[str, Any]
) -> None:
    hub = _hub_with_previews(tmp_path)

    response = client_for(hub).patch("/preview-settings", json=patch)

    assert response.status_code == 422
    assert hub.take(paused=False) == []


def test_a_trainer_without_preview_settings_answers_404() -> None:
    client = client_for(ControlHub())

    assert client.get("/preview-settings").status_code == 404
    assert client.patch("/preview-settings", json={"every": 5}).status_code == 404


def test_swagger_and_the_schema_are_served() -> None:
    client = client_for(ControlHub())

    assert "swagger" in client.get("/docs").text.lower()
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/status", "/commands/{action}", "/commands", "/commands/{command_id}", "/preview-settings"} <= set(paths)
