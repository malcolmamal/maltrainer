"""The numbers the control API reports about a running trainer, read on the training thread at a step boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from maltrainer.utils.train_monitor import TrainMonitor


def learning_rate(optimizer: Any) -> float:
    """The rate the progress line prints: Automagic3's average, Prodigy's multiplier x d, or the plain rate."""
    inner = getattr(optimizer, "optimizer", optimizer)  # accelerate's wrapper does not forward custom methods
    if hasattr(inner, "get_avg_learning_rate"):
        return float(inner.get_avg_learning_rate())
    group = optimizer.param_groups[0]
    return float(group["lr"] * group["d"]) if group.get("d") is not None else float(group["lr"])


@dataclass
class RunView:
    """What to read from a trainer: its ``optimizer``, ``epoch`` (when it has one), the planned step count in
    ``run_metadata`` and the training monitor (given, or the trainer's ``monitor`` attribute)."""

    host: Any
    monitor: TrainMonitor | None = None

    def stats(self, step: int) -> dict[str, Any]:
        total = self.host.run_metadata.get("planned_total_steps")
        monitor = self.monitor or getattr(self.host, "monitor", None)
        window = monitor.snapshot() if monitor is not None else None
        seconds = window["seconds_per_step"] if window else None
        epoch = getattr(self.host, "epoch", None)
        return {
            "step": step,
            "total_steps": None if total is None else int(total),
            "percent": round(100 * step / total, 1) if total else None,
            "epoch": None if epoch is None else int(epoch) + 1,
            "learning_rate": learning_rate(self.host.optimizer),
            "eta_seconds": None if seconds is None or not total else round(seconds * max(total - step, 0), 1),
            "window": window,
        }
