"""What a LoRA run tells the control API at a step boundary: the kind, the epoch and the monitor's window."""

from __future__ import annotations

from typing import Any

import pytest

from maltrainer.control.stats import RunView
from maltrainer.training import loop

from .fakes import ScriptedKeys
from .runs import run_lora
from .world import Workspace


def _stats_at_each_step(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the loop's key factory with scripted keys that read the run's stats at every step boundary."""
    seen: list[dict[str, Any]] = []

    def make_keys(_factory: object, _config: object, view: RunView, **_extra: object) -> ScriptedKeys:
        keys = ScriptedKeys([])
        handle = keys.handle

        def read_then_handle(step: int, save: Any) -> bool:
            seen.append(view.stats(step))
            return handle(step, save)

        keys.handle = read_then_handle  # type: ignore[method-assign]
        return keys

    monkeypatch.setattr(loop, "make_keys", make_keys)
    return seen


def test_a_lora_run_reports_its_kind_epoch_and_monitor_window(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    seen = _stats_at_each_step(monkeypatch)

    run_lora(workspace.lora_config(train={"max_steps": 2, "monitor_every": 100}))

    first = seen[0]
    assert (first["kind"], first["step"], first["total_steps"], first["epoch"]) == ("lora", 1, 2, 1)
    assert first["window"]["window_steps"] == 1
    assert first["window"]["window_loss"] is not None
    assert first["window"]["seconds_per_step"] is not None
    assert first["eta_seconds"] is not None


def test_the_monitor_logs_its_window_every_monitor_every_steps_for_a_lora_run(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)

    with caplog.at_level("INFO"):
        run_lora(workspace.lora_config(train={"max_steps": 2, "monitor_every": 1}))

    assert caplog.text.count("[monitor] step") == 2
