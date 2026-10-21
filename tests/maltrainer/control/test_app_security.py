from __future__ import annotations

import pytest

from maltrainer.control.hub import ControlHub

from .helpers import client_for


def test_with_a_token_every_call_needs_it() -> None:
    hub = ControlHub()
    client = client_for(hub, token="s3cret")

    assert client.get("/status").status_code == 401
    assert client.post("/commands/save").status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert hub.take(paused=False) == []


@pytest.mark.parametrize(
    ("headers", "allowed"),
    [
        ({"host": "127.0.0.1:8765"}, True),
        ({"host": "localhost:8765"}, True),
        ({"host": "[::1]:8765"}, True),
        ({"host": "evil.example:8765"}, False),
        ({"host": "127.0.0.1:8765", "origin": "http://127.0.0.1:8765"}, True),
        ({"host": "127.0.0.1:8765", "origin": "http://evil.example"}, False),
        ({"host": "127.0.0.1:8765", "origin": "null"}, False),
    ],
)
def test_a_page_on_another_site_or_a_rebound_name_cannot_steer_the_run(headers: dict[str, str], allowed: bool) -> None:
    hub = ControlHub()
    client = client_for(hub)

    response = client.post("/commands/save", headers=headers)

    assert response.status_code == (202 if allowed else 403)
    assert bool(hub.take(paused=False)) is allowed


def test_with_a_token_the_api_may_be_reached_under_any_name_but_never_from_another_origin() -> None:
    hub = ControlHub()
    client = client_for(hub, token="s3cret")
    bearer = {"Authorization": "Bearer s3cret"}

    named = client.post("/commands/save", headers={**bearer, "host": "192.168.1.5:8765"})
    foreign = client.post("/commands/pause", headers={**bearer, "origin": "http://evil.example"})

    assert (named.status_code, foreign.status_code) == (202, 403)
    assert [p.record.command.value for p in hub.take(paused=False)] == ["save"]
