"""Data side of scripts/qwen_2_1_bench_report.py: training logs, adapter magnitude and per-run dynamics."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

LOG_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - progress - INFO - Step: (\d+)/\d+ \[\d+%\], "
    r"Loss: ([0-9.]+), Epoch: \d+/\d+, LR: ([0-9.eE+-]+)"
)
COLORS = {"adamw": "#2f6fdf", "prodigy": "#d9480f", "automagic3": "#2b8a3e", "base": "#868e96"}


def color(name: str) -> str | None:
    """Colour by optimizer, so prefixed run names (rhea_adamw) keep the same palette."""
    return next((c for key, c in COLORS.items() if name == key or name.endswith("_" + key)), None)


def parse_log(path: Path) -> dict[str, Any]:
    steps, losses, lrs, times = [], [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_LINE.match(line)
        if m:
            times.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            steps.append(int(m.group(2)))
            losses.append(float(m.group(3)))
            lrs.append(float(m.group(4)))
    if not steps:
        raise SystemExit(f"no progress lines in {path}")
    seconds = (times[-1] - times[0]).total_seconds()
    return {
        "step": steps,
        "loss": losses,
        "lr": lrs,
        "sec_per_step": seconds / max(1, steps[-1] - steps[0]),
        "wall_minutes": seconds / 60,
    }


def ema(values: list[float], alpha: float = 0.02) -> np.ndarray:
    out, acc = [], values[0]
    for v in values:
        acc = alpha * v + (1 - alpha) * acc
        out.append(acc)
    return np.array(out)


def adapter_norm(path: Path) -> float:
    """Mean over layers of ||B @ A||_F * alpha/rank, via the rank-sized Gram trick."""
    norms = []
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
        for key in keys:
            if not key.endswith(".lora_A.weight"):
                continue
            a = f.get_tensor(key).float()
            b = f.get_tensor(key.replace(".lora_A.", ".lora_B.")).float()
            alpha_key = key.replace(".lora_A.weight", ".alpha")
            alpha = f.get_tensor(alpha_key).item() if alpha_key in keys else a.shape[0]
            frob2 = torch.trace((b.T @ b) @ (a @ a.T)).clamp_min(0)
            norms.append(float(frob2.sqrt()) * alpha / a.shape[0])
    return float(np.mean(norms))


def step_num(step: str, final_step: int) -> int:
    return final_step if step == "final" else int(step)


def checkpoint_step(path: Path) -> str:
    m = re.search(r"_step_(\d+)$", path.stem)
    return m.group(1) if m else "final"


def collect_dynamics(runs: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Per run: the log's series and speed, loss/LR summaries and ||dW|| per checkpoint."""
    dynamics: dict[str, dict[str, Any]] = {}
    for name, raw_dir in runs.items():
        run_dir = Path(raw_dir)
        log = parse_log(run_dir / "training.log")
        ckpts = sorted(
            run_dir.glob("*.safetensors"),
            key=lambda p: 10**9 if checkpoint_step(p) == "final" else int(checkpoint_step(p)),
        )
        final_step = log["step"][-1]
        dynamics[name] = {
            **log,
            "loss_first200": float(np.mean(log["loss"][:200])),
            "loss_last200": float(np.mean(log["loss"][-200:])),
            "lr_final": log["lr"][-1],
            "lr_max": max(log["lr"]),
            "dw": [(step_num(checkpoint_step(c), final_step), adapter_norm(c)) for c in ckpts],
        }
        print(
            f"{name:>11}: {log['sec_per_step']:.3f} s/step, {log['wall_minutes']:.1f} min, "
            f"loss {dynamics[name]['loss_first200']:.4f} -> {dynamics[name]['loss_last200']:.4f}, "
            f"LR final {log['lr'][-1]:.2e} (max {max(log['lr']):.2e}), ||dW|| final {dynamics[name]['dw'][-1][1]:.3f}"
        )
    return dynamics
