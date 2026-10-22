from __future__ import annotations

import json
import logging
import socket
import time
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from maltrainer.control.keys import ControlledKeys
from maltrainer.control.launch import make_keys
from maltrainer.utils.keyboard import KeyCommands
from maltrainer.utils.previews import Previews

from .helpers import Saver, make_view

CONFIG: dict[str, Any] = {"control_api": {"enabled": True, "port": 0}}


def _call(keys: ControlledKeys, path: str, method: str = "GET", body: dict[str, Any] | None = None) -> Any:
    parts = urlsplit(keys.url)
    connection = HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        data = None if body is None else json.dumps(body)
        connection.request(method, path, body=data, headers={"Content-Type": "application/json"})
        return json.loads(connection.getresponse().read())
    finally:
        connection.close()


@pytest.fixture
def served() -> Iterator[list[ControlledKeys]]:
    made: list[ControlledKeys] = []
    yield made
    for keys in made:
        keys.close()


def _make(served: list[ControlledKeys], **extra: Any) -> ControlledKeys:
    keys = make_keys(KeyCommands, CONFIG, make_view(), **extra)
    assert isinstance(keys, ControlledKeys)
    served.append(keys)
    return keys


def test_without_the_section_the_plain_keys_are_built_with_only_the_options_given() -> None:
    built: list[dict[str, Any]] = []

    def factory(**options: Any) -> KeyCommands:
        built.append(options)
        return KeyCommands(**options)

    plain = make_keys(factory, {}, make_view(), menu=lambda: None)
    make_keys(factory, {"control_api": {"enabled": False}}, make_view())

    assert type(plain) is KeyCommands
    assert [sorted(options) for options in built] == [["menu"], []]


def test_the_api_serves_status_and_a_save_and_stop_over_http_stops_the_loop(served: list[ControlledKeys]) -> None:
    keys = _make(served)
    keys.handle(3, Saver())
    status = _call(keys, "/status")
    assert (status["state"], status["step"], status["total_steps"]) == ("running", 3, 100)

    command = _call(keys, "/commands/save-and-stop", "POST")
    save = Saver()
    stopped = keys.handle(4, save)

    assert command["state"] == "pending"
    assert stopped is True
    assert save.steps == [4]
    assert _call(keys, f"/commands/{command['id']}")["state"] == "done"
    assert _call(keys, "/status")["state"] == "stopping"


def test_preview_settings_travel_the_whole_way(served: list[ControlledKeys], tmp_path: Path) -> None:
    previews = Previews({"every": 100, "prompts": ["a", "b"]}, tmp_path)
    keys = _make(served, previews=previews, row_labels=lambda: ["raw"])
    assert _call(keys, "/preview-settings")["rows"] == {"raw": True}

    _call(keys, "/preview-settings", "PATCH", {"every": 40, "prompts": [{"slot": 0, "enabled": False}]})
    keys.handle(1, Saver())

    assert previews.every == 40
    assert previews.active == ["b"]
    assert _call(keys, "/preview-settings")["every"] == 40


def test_a_port_in_use_is_logged_and_the_run_gets_its_plain_keys(caplog: pytest.LogCaptureFixture) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        port = busy.getsockname()[1]
        config = {"control_api": {"enabled": True, "port": port}}

        with caplog.at_level(logging.ERROR, logger="maltrainer.control.launch"):
            keys = make_keys(KeyCommands, config, make_view())

    assert type(keys) is KeyCommands
    assert f"could not start on 127.0.0.1:{port}" in caplog.text
    assert "training goes on without it" in caplog.text


def test_a_bad_section_stops_the_start_with_the_reason() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        make_keys(KeyCommands, {"control_api": {"enabled": True, "prot": 1}}, make_view())


def test_the_server_stops_after_the_run_ends(served: list[ControlledKeys]) -> None:
    keys = _make(served)
    assert _call(keys, "/status")["state"] == "running"

    keys.close()
    assert _call(keys, "/status")["state"] == "finished"

    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            _call(keys, "/status")
        except OSError:
            return
        time.sleep(0.2)
    pytest.fail("the control API kept serving after the run ended")
