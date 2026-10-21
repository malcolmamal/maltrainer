"""A client of a training run's control API, driven from a thread while the real trainer trains on the tiny world."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from http.client import HTTPConnection
from typing import Any
from urllib.parse import urlsplit

from maltrainer.control.keys import ControlledKeys
from maltrainer.utils.keyboard import KeyCommands

CONTROL = {"control_api": {"enabled": True, "port": 0}}


class Driver:
    """Wrap a trainer module's ``make_preview_host_keys`` with ``driver.wrap``: the run pauses at its first step
    boundary, and ``script`` (given the driver) steers it over HTTP from there, ending with a save-and-stop."""

    def __init__(self, script: Callable[[Driver], None]) -> None:
        self.script = script
        self.keys: ControlledKeys | None = None
        self.seen: dict[str, Any] = {}
        self.error: Exception | None = None
        self.thread: threading.Thread | None = None

    def wrap(self, real: Callable[..., KeyCommands]) -> Callable[..., KeyCommands]:
        def made(*args: Any, **kwargs: Any) -> KeyCommands:
            keys = real(*args, **kwargs)
            assert isinstance(keys, ControlledKeys)
            self.keys = keys
            self.call("POST", "/commands/pause")
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            return keys

        return made

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        assert self.keys is not None
        parts = urlsplit(self.keys.url)
        connection = HTTPConnection(parts.hostname, parts.port, timeout=30)
        try:
            data = None if body is None else json.dumps(body)
            connection.request(method, path, body=data, headers={"Content-Type": "application/json"})
            return json.loads(connection.getresponse().read())
        finally:
            connection.close()

    def until(self, check: Callable[[], bool], what: str, seconds: float = 60) -> None:
        deadline = time.time() + seconds
        while not check():
            if time.time() > deadline:
                raise TimeoutError(f"waited {seconds:.0f} s for {what}")
            time.sleep(0.05)

    def settled(self, command: dict[str, Any]) -> dict[str, Any]:
        """The command once it is no longer pending."""
        self.until(lambda: self.call("GET", f"/commands/{command['id']}")["state"] != "pending", "a command to settle")
        return dict(self.call("GET", f"/commands/{command['id']}"))

    def _run(self) -> None:
        try:
            self.until(lambda: self.call("GET", "/status")["state"] == "paused", "the pause")
            self.script(self)
        except (AssertionError, TimeoutError, OSError, ValueError, KeyError) as error:  # the test reports it
            self.error = error
        finally:  # whatever happened, the run must end
            with contextlib.suppress(OSError, ValueError):
                self.call("POST", "/commands/save-and-stop")

    def finish(self) -> None:
        """After the run: wait for the thread and raise what went wrong in it."""
        if self.thread is not None:
            self.thread.join(timeout=60)
        if self.error is not None:
            raise self.error
