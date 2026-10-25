"""What a Krea 2 finetune tells the control API: readable before the first window comes up, and the active window's
rate once one has (the finetune's optimizers live in the window rotator, not in ``optimizer``)."""

from __future__ import annotations

from typing import Any

import pytest

from maltrainer.control import launch
from maltrainer.control.stats import RunView

from .fakes import ScriptedKeys
from .runs import run_finetune
from .world import Workspace


def _stats(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The run's stats when its keys are made (before step 1, as the control API publishes) and at every step."""
    seen: list[dict[str, Any]] = []

    def make_keys(_factory: object, _config: object, view: RunView, **_extra: object) -> ScriptedKeys:
        seen.append(view.stats(0))
        keys = ScriptedKeys([], None, None)
        handle = keys.handle

        def read_then_handle(step: int, save: Any) -> bool:
            seen.append(view.stats(step))
            return handle(step, save)

        keys.handle = read_then_handle  # type: ignore[method-assign]
        return keys

    monkeypatch.setattr(launch, "make_keys", make_keys)
    return seen


def test_the_stats_read_before_any_window_and_report_the_active_windows_rate_after(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    seen = _stats(monkeypatch)

    run_finetune(workspace.finetune_config(train={"max_steps": 3}))

    start, steps = seen[0], seen[1:]
    assert (start["kind"], start["step"], start["learning_rate"]) == ("finetune", 0, None)
    assert [s["step"] for s in steps] == [1, 2, 3]
    assert all(s["learning_rate"] == pytest.approx(1e-3) for s in steps)  # the pinned rate of the test config
