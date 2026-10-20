"""Charts and comparison grids for scripts/qwen_2_1_bench_report.py. ``plt`` is passed in (the script selects Agg)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from qwen_2_1_bench_report_data import color, ema, step_num


def plot_dynamics(plt: Any, dynamics: dict[str, dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for name, d in dynamics.items():
        c = color(name)
        axes[0].plot(d["step"], ema(d["loss"]), color=c, label=name, lw=1.6)
        axes[1].plot(d["step"], d["lr"], color=c, label=name, lw=1.6)
        xs, ys = zip(*d["dw"], strict=False)
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


def plot_image_metrics(
    plt: Any, summary: dict[str, Any], dynamics: dict[str, dict[str, Any]], runs: dict[str, str], out: Path
) -> None:
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
    for ax, (key, title, refs) in zip(axes, panels, strict=False):
        for name in runs:
            pts = sorted(
                (step_num(r["step"], final_steps[name]), r[key])
                for r in rows
                if r["run"] == name and r.get(key) is not None
            )
            if pts:
                ax.plot(*zip(*pts, strict=False), "o-", color=color(name), label=name, lw=1.6, ms=4)
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


def write_grids(
    manifest: dict[str, Any], runs: dict[str, str], bench: Path, out: Path, grid_steps: list[str], grid_seed: int
) -> None:
    for prompt in manifest["prompts"]:
        pid = prompt["id"]
        cols = ["base", *grid_steps]
        cell, pad, head, left = 256, 6, 24, 110
        sheet = Image.new("RGB", (left + len(cols) * (cell + pad), head + len(runs) * (cell + pad)), "white")
        draw = ImageDraw.Draw(sheet)
        for ci, step in enumerate(cols):
            draw.text(
                (left + ci * (cell + pad) + 4, 6), "base (no LoRA)" if step == "base" else f"step {step}", fill="black"
            )
        for r, name in enumerate(runs):
            y = head + r * (cell + pad)
            draw.text((6, y + cell // 2), name, fill="black")
            for ci, step in enumerate(cols):
                rel = (Path("base") if step == "base" else Path(name) / step) / f"{pid}_s{grid_seed}.png"
                path = bench / rel
                if path.exists():
                    sheet.paste(
                        Image.open(path).convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS),
                        (left + ci * (cell + pad), y),
                    )
        sheet.save(out / f"grid_{pid}.png")
