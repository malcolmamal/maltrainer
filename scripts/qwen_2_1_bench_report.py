#!/usr/bin/env python3
"""Charts and tables for an optimizer benchmark: training dynamics + scored image metrics.

Runs in ComfyUI's venv (matplotlib, torch, safetensors). Reads:
  * each run's training.log            loss, effective LR, wall-clock speed
  * each run's checkpoints             adapter magnitude ||dW|| per snapshot
  * <bench>/summary.json, images       from qwen_2_1_bench_generate.py / _score.py

Writes into <bench>/report/:
  training_dynamics.png   loss (EMA), effective learning rate, ||dW|| vs step
  image_metrics.png       likeness, adherence, diversity, memorization vs step
  grid_<prompt>.png       optimizers x checkpoints, one seed
  report.json             everything above as numbers

Usage:
    C:\\Development\\ComfyUI\\venv\\Scripts\\python.exe scripts\\qwen_2_1_bench_report.py --bench samples\\bench \\
        --runs adamw=output\\bench_adamw prodigy=output\\bench_prodigy automagic3=output\\bench_automagic3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qwen_2_1_bench_report_data import (  # noqa: F401  (re-exported)
    COLORS,
    LOG_LINE,
    adapter_norm,
    checkpoint_step,
    collect_dynamics,
    color,
    ema,
    parse_log,
    step_num,
)
from qwen_2_1_bench_report_plots import plot_dynamics, plot_image_metrics, write_grids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="name=output_dir pairs")
    parser.add_argument("--grid-steps", nargs="+", default=["500", "1000", "1500", "2000"])
    parser.add_argument("--grid-seed", type=int, default=1)
    args = parser.parse_args()
    bench = Path(args.bench)
    out = bench / "report"
    out.mkdir(exist_ok=True)
    runs = dict(r.split("=", 1) for r in args.runs)
    summary = json.loads((bench / "summary.json").read_text())
    manifest = json.loads((bench / "manifest.json").read_text())

    dynamics = collect_dynamics(runs)
    plot_dynamics(plt, dynamics, out)
    plot_image_metrics(plt, summary, dynamics, runs, out)
    write_grids(manifest, runs, bench, out, args.grid_steps, args.grid_seed)

    report = {
        "dynamics": {n: {k: v for k, v in d.items() if k not in ("step", "loss", "lr")} for n, d in dynamics.items()},
        "summary": summary,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote charts and grids to {out}")


if __name__ == "__main__":
    main()
