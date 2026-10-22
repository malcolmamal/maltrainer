"""A synthetic benchmark (images, manifest, training set) and colour-based stand-ins for the scoring models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import qwen_2_1_bench_score as score

RED, ORANGE, YELLOW, BLACK = (255, 0, 0), (200, 50, 0), (100, 100, 0), (0, 0, 0)
PROMPTS = [{"id": "p1", "checks": ["Is it red?", "Is it bright?"]}, {"id": "p2"}]
# (run, step) -> colour of its images; BLACK has no "face"
COLOURS = {("base", "base"): YELLOW, ("adamw", "5"): ORANGE, ("adamw", "final"): RED}
SAMPLES = [("p1", 1), ("p1", 2), ("p2", 1)]


def unit(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    return array / (np.linalg.norm(array) + 1e-12)


def colour_of(image: Image.Image) -> tuple[int, ...]:
    """The colour of an image's top-left pixel (the images here are solid)."""
    return tuple(int(v) for v in np.asarray(image.convert("RGB"))[0, 0])


class FakeClip:
    """CLIP-H stand-in: the unit vector of an image's colour (plus a constant), so equal colours are identical."""

    def __init__(self, device: str) -> None:
        self.device = device

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        return np.stack([unit([*colour_of(image), 255]) for image in images])


class FakeFaces:
    """ArcFace stand-in: a face is the unit vector of the colour; a black image shows no face."""

    def __init__(self, clip: FakeClip, root: str | None, name: str) -> None:
        self.backend = "fake"
        self.clip, self.args = clip, (root, name)

    def __call__(self, images: list[Image.Image]) -> list[np.ndarray | None]:
        return [None if colour_of(i) == BLACK else unit(colour_of(i)) for i in images]


class FakeJudge:
    """Qwen3-VL stand-in: P(yes) is the image's red level for the first question, the green level plus 0.1 else."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.asked: list[str] = []

    def __call__(self, image: Image.Image, question: str) -> float:
        self.asked.append(question)
        red, green, _ = colour_of(image)
        return red / 255 if question == "Is it red?" else green / 255 + 0.1


def make_bench(root: Path, colours: dict[tuple[str, str], tuple[int, ...]] | None = None, faces: int = 3) -> Path:
    """<root>/bench with manifest.json and images, and <root>/train with ``faces`` red photos. Returns the bench."""
    bench, train = root / "bench", root / "train"
    train.mkdir()
    for i in range(faces):
        Image.new("RGB", (16, 16), RED).save(train / f"t{i}.png")
    records = []
    for (run, step), colour in (colours or COLOURS).items():
        folder = Path("base") if run == "base" else Path(run) / step
        (bench / folder).mkdir(parents=True, exist_ok=True)
        for prompt, seed in SAMPLES:
            path = folder / f"{prompt}_s{seed}.png"
            Image.new("RGB", (16, 16), colour).save(bench / path)
            records.append({"run": run, "step": step, "prompt": prompt, "seed": seed, "path": str(path)})
    manifest = {"prompts": PROMPTS, "dataset": str(train), "images": records}
    (bench / "manifest.json").write_text(json.dumps(manifest))
    return bench


def run_score(monkeypatch: pytest.MonkeyPatch, bench: Path, *flags: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the script's main on ``bench`` with the stand-in models; return scores.json and summary.json."""
    monkeypatch.setattr(score, "ClipVisionH", FakeClip)
    monkeypatch.setattr(score, "FaceEmbedder", FakeFaces)
    monkeypatch.setattr(score, "YesNoJudge", FakeJudge)
    monkeypatch.setattr("sys.argv", ["score", "--bench", str(bench), "--device", "cpu", *flags])
    score.main()
    return json.loads((bench / "scores.json").read_text()), json.loads((bench / "summary.json").read_text())


def summary_row(summary: dict[str, Any], run: str, step: str) -> dict[str, Any]:
    return next(r for r in summary["rows"] if (r["run"], r["step"]) == (run, step))


def like(colour: tuple[int, ...]) -> float:
    """The likeness the stand-in face model gives a colour against the red training set."""
    return float(unit(colour) @ unit(RED))
