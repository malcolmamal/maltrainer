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
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open

LOG_LINE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - progress - INFO - Step: (\d+)/\d+ \[\d+%\], "
    r"Loss: ([0-9.]+), Epoch: \d+/\d+, LR: ([0-9.eE+-]+)"
)
COLORS = {"adamw": "#2f6fdf", "prodigy": "#d9480f", "automagic3": "#2b8a3e", "base": "#868e96"}


def color(name: str) -> str | None:
    """Colour by optimizer, so prefixed run names (rhea_adamw) keep the same palette."""
    return next((c for key, c in COLORS.items() if name == key or name.endswith("_" + key)), None)


def parse_log(path: Path) -> dict:
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

    # ---- training dynamics --------------------------------------------------------------
    dynamics = {}
    for name, run_dir in runs.items():
        run_dir = Path(run_dir)
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

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, d in dynamics.items():
        c = color(name)
        axes[0].plot(d["step"], ema(d["loss"]), color=c, label=name, lw=1.6)
        axes[1].plot(d["step"], d["lr"], color=c, label=name, lw=1.6)
        xs, ys = zip(*d["dw"])
        axes[2].plot(xs, ys, "o-", color=c, label=name, lw=1.6, ms=4)
    axes[0].set(title="Training loss (EMA, alpha 0.02)", xlabel="step", ylabel="MSE")
    axes[1].set(title="Effective learning rate", xlabel="step", ylabel="LR", yscale="log")
    axes[2].set(title="Adapter magnitude  mean ||B·A||·α/r", xlabel="step", ylabel="||ΔW||")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_dynamics.png", dpi=110)
    plt.close(fig)

    # ---- image metrics ---------------------------------------------------------------------
    ref = summary["reference"]
    rows = [r for r in summary["rows"] if r["run"] != "base"]
    base_row = next(r for r in summary["rows"] if r["run"] == "base")
    final_steps = {n: d["step"][-1] for n, d in dynamics.items()}
    panels = [
        ("likeness_norm", "Likeness (0 = base model, 1 = real photo)", [(0, "base"), (1, "real photos")]),
        ("adherence", "Prompt adherence vs base (VLM judge, Δ P(yes))", [(0, "base")]),
        ("diversity", "Seed diversity (1 − mean CLIP sim)", [(base_row["diversity"], "base")]),
        (
            "memorization",
            "Memorization (max CLIP sim to a training image)",
            [(base_row["memorization"], "base"), (ref["training_pairwise_clip"], "train pairwise")],
        ),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
    for ax, (key, title, refs) in zip(axes, panels):
        for name in runs:
            pts = sorted(
                (step_num(r["step"], final_steps[name]), r[key])
                for r in rows
                if r["run"] == name and r.get(key) is not None
            )
            if pts:
                ax.plot(*zip(*pts), "o-", color=color(name), label=name, lw=1.6, ms=4)
        for y, label in refs:
            if y is not None:
                ax.axhline(y, color="#adb5bd", ls="--", lw=1)
                ax.annotate(label, (ax.get_xlim()[0], y), fontsize=8, color="#868e96", va="bottom")
        ax.set(title=title, xlabel="step")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "image_metrics.png", dpi=110)
    plt.close(fig)

    # ---- comparison grids ---------------------------------------------------------------------
    for prompt in manifest["prompts"]:
        pid = prompt["id"]
        cols = ["base"] + args.grid_steps
        cell, pad, head, left = 256, 6, 24, 110
        sheet = Image.new("RGB", (left + len(cols) * (cell + pad), head + len(runs) * (cell + pad)), "white")
        draw = ImageDraw.Draw(sheet)
        for c, step in enumerate(cols):
            draw.text(
                (left + c * (cell + pad) + 4, 6), "base (no LoRA)" if step == "base" else f"step {step}", fill="black"
            )
        for r, name in enumerate(runs):
            y = head + r * (cell + pad)
            draw.text((6, y + cell // 2), name, fill="black")
            for c, step in enumerate(cols):
                rel = (Path("base") if step == "base" else Path(name) / step) / f"{pid}_s{args.grid_seed}.png"
                path = bench / rel
                if path.exists():
                    sheet.paste(
                        Image.open(path).convert("RGB").resize((cell, cell), Image.LANCZOS),
                        (left + c * (cell + pad), y),
                    )
        sheet.save(out / f"grid_{pid}.png")

    report = {
        "dynamics": {n: {k: v for k, v in d.items() if k not in ("step", "loss", "lr")} for n, d in dynamics.items()},
        "summary": summary,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote charts and grids to {out}")


if __name__ == "__main__":
    main()
