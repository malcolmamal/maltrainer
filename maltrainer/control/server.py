"""Runs the control API in a daemon thread of the trainer's process, bound before training goes on."""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI

from maltrainer.control.settings import ControlSettings

LINGER_SECONDS = 1.0  # after the run ends: long enough for a client to read "finished"


@dataclass
class RunningServer:
    url: str
    server: uvicorn.Server
    thread: threading.Thread

    def stop(self, linger: float = LINGER_SECONDS) -> None:
        """Stop serving after ``linger`` seconds, without waiting for it."""

        def later() -> None:
            time.sleep(linger)
            self.server.should_exit = True

        threading.Thread(target=later, daemon=True).start()


def start_server(app: FastAPI, settings: ControlSettings) -> RunningServer:
    """Bind first, so a port in use is an OSError here and not a thread that quietly died; then serve in a thread."""
    family = socket.AF_INET6 if ":" in settings.host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((settings.host, settings.port))
    except OSError:
        sock.close()
        raise
    host, port = sock.getsockname()[:2]
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, name="control-api", daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and thread.is_alive() and time.time() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise OSError("the control API server did not start")
    shown = f"[{host}]" if ":" in host else host
    return RunningServer(f"http://{shown}:{port}", server, thread)
