"""A trainer stand-in for the control tests: what ``RunView`` reads, built from real parts (an optimizer, a monitor)."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import torch

from maltrainer.control.app import create_app
from maltrainer.control.hub import Capabilities, ControlHub
from maltrainer.control.keys import ControlledKeys, PreviewTarget
from maltrainer.control.settings import ControlSettings
from maltrainer.control.stats import RunView
from maltrainer.utils.train_monitor import TrainMonitor
from tests.backend.fakes import TestClient


def make_view(total: int = 100, epoch: int | None = 0) -> RunView:
    parameter = torch.nn.Parameter(torch.zeros(1))
    host = SimpleNamespace(
        run_metadata={"planned_total_steps": total},
        optimizer=torch.optim.SGD([parameter], lr=0.01),
        monitor=TrainMonitor(every=10),
    )
    if epoch is not None:
        host.epoch = epoch
    return RunView(host)


class Saver:
    """The trainer's save_checkpoint: records the steps, and can be told that the save did not happen."""

    def __init__(self, result: bool | None = None) -> None:
        self.steps: list[int] = []
        self.result = result

    def __call__(self, step: int) -> bool | None:
        self.steps.append(step)
        return self.result


def controlled_keys(
    hub: ControlHub | None = None,
    gallery: Callable[[], None] | None = None,
    previews: PreviewTarget | None = None,
) -> tuple[ControlledKeys, ControlHub]:
    hub = hub or ControlHub(Capabilities(gallery=gallery is not None, preview_settings=previews is not None))
    return ControlledKeys(hub, make_view(), previews, gallery=gallery), hub


def client_for(hub: ControlHub, token: str = "") -> TestClient:
    """A test client of the API over ``hub``, calling from the loopback address the settings name."""
    app = create_app(hub, ControlSettings(host="127.0.0.1", port=8765, token=token))
    return TestClient(app, base_url="http://127.0.0.1:8765")
