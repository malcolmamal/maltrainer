"""A stand-in for matplotlib and a synthetic benchmark folder for the qwen_2_1_bench_report tests."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

STAMPS = ["2026-01-01 10:00:00", "2026-01-01 10:00:10", "2026-01-01 10:00:30"]


def log_line(stamp: str, step: int, loss: float, lr: float) -> str:
    return f"{stamp},123 - progress - INFO - Step: {step}/3 [{step * 33}%], Loss: {loss}, Epoch: 1/1, LR: {lr:.2e}"


def write_log(path: Path, losses: list[float]) -> Path:
    lines = ["2026-01-01 10:00:00,001 - noise - INFO - not a progress line"]
    lines += [log_line(STAMPS[i], i + 1, loss, 1e-3 / (i + 1)) for i, loss in enumerate(losses)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_adapter(path: Path, scale: float) -> Path:
    down, up = torch.eye(2), torch.diag(torch.tensor([3.0, 4.0])) * scale
    save_file({"m.lora_A.weight": down, "m.lora_B.weight": up, "m.alpha": torch.tensor(4.0)}, str(path))
    return path


def make_bench(root: Path, runs: tuple[str, ...] = ("adamw", "prodigy")) -> Path:
    """<root>/bench with summary, manifest and images, plus <root>/out_<run>/ with a log and two checkpoints."""
    bench = root / "bench"
    bench.mkdir()
    base = {
        "run": "base",
        "step": "base",
        "likeness_norm": 0.0,
        "adherence": 0.0,
        "diversity": 0.4,
        "memorization": 0.5,
    }
    rows = [base]
    for run in runs:
        for step in ("5", "final"):
            metrics = {"likeness_norm": 0.5, "adherence": -0.1, "diversity": 0.3, "memorization": 0.6}
            rows.append({"run": run, "step": step, **metrics})
    (bench / "summary.json").write_text(json.dumps({"reference": {"training_pairwise_clip": 0.7}, "rows": rows}))
    (bench / "manifest.json").write_text(json.dumps({"prompts": [{"id": "portrait"}, {"id": "knight"}]}))
    (bench / "base").mkdir()
    Image.new("RGB", (8, 8), "red").save(bench / "base" / "portrait_s1.png")
    for run in runs:
        (bench / run / "5").mkdir(parents=True)
        Image.new("RGB", (8, 8), "blue").save(bench / run / "5" / "portrait_s1.png")
        out = root / f"out_{run}"
        out.mkdir()
        write_log(out / "training.log", [1.0, 0.5, 0.25])
        write_adapter(out / f"{run}_step_5.safetensors", 1.0)
        write_adapter(out / f"{run}.safetensors", 2.0)
    return bench


@contextmanager
def fake_matplotlib(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[MagicMock, ModuleType]]:
    """``qwen_2_1_bench_report`` imported against a mock ``matplotlib.pyplot`` (matplotlib is not installed)."""
    plt = MagicMock()
    figures: list[Any] = []
    axes_by_figure: list[list[Any]] = []

    def subplots(_rows: int, cols: int, **_options: Any) -> tuple[Any, list[Any]]:
        axes = [MagicMock() for _ in range(cols)]
        for ax in axes:
            ax.get_xlim.return_value = (0.0, 1.0)
        figures.append(MagicMock())
        axes_by_figure.append(axes)
        return figures[-1], axes

    plt.subplots.side_effect = subplots
    plt.figures = figures
    plt.axes_by_figure = axes_by_figure
    matplotlib = MagicMock()
    matplotlib.pyplot = plt
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)
    try:
        yield plt, importlib.import_module("qwen_2_1_bench_report")
    finally:
        sys.modules.pop("qwen_2_1_bench_report", None)
