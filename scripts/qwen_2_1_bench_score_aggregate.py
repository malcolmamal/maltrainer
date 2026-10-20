"""Reference faces, per-image scoring and per-(run, step) aggregates for scripts/qwen_2_1_bench_score.py."""

from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from qwen_2_1_bench_score_models import l2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_training_images(dataset: Path) -> list[Image.Image]:
    paths = sorted(p for p in dataset.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return [Image.open(p).convert("RGB") for p in paths]


def step_order(step: str) -> int:
    return 10**9 if step == "final" else (-1 if step == "base" else int(step))


def train_reference(clip: Any, faces: Any, train_images: list[Image.Image]) -> dict[str, Any]:
    """Training faces, their centroid, the likeness ceiling and the training set's CLIP embeddings."""
    train_face_list = [f for f in faces(train_images) if f is not None]
    if len(train_face_list) < 3:
        raise SystemExit(f"only {len(train_face_list)} faces found in the training set")
    train_faces = np.stack(train_face_list)
    centroid = l2(train_faces.mean(0))
    ceiling = float(
        np.mean([train_faces[i] @ l2(np.delete(train_faces, i, 0).mean(0)) for i in range(len(train_faces))])
    )
    train_clip = clip(train_images)
    train_pairwise = float((train_clip @ train_clip.T)[~np.eye(len(train_clip), dtype=bool)].mean())
    print(f"training faces {len(train_faces)}/{len(train_images)}, likeness ceiling {ceiling:.4f}")
    return {
        "faces": len(train_faces),
        "centroid": centroid,
        "ceiling": ceiling,
        "clip": train_clip,
        "pairwise": train_pairwise,
    }


def score_images(
    records: list[dict[str, Any]],
    image_clip: np.ndarray,
    image_faces: list[Any],
    reference: dict[str, Any],
) -> None:
    """Face likeness to the training centroid and memorization (max CLIP similarity to a training image)."""
    for r, emb, face in zip(records, image_clip, image_faces, strict=False):
        r["face_found"] = face is not None
        r["likeness"] = float(face @ reference["centroid"]) if face is not None else None
        r["memorization"] = float((reference["clip"] @ emb).max())


def judge_records(
    judge: Any, records: list[dict[str, Any]], images: list[Image.Image], checks: dict[str, list[str]]
) -> None:
    for i, (r, image) in enumerate(zip(records, images, strict=False)):
        qs = checks.get(r["prompt"], [])
        r["checks"] = {q: judge(image, q) for q in qs}
        r["judge"] = float(np.mean(list(r["checks"].values()))) if qs else None
        if (i + 1) % 25 == 0:
            print(f"  judged {i + 1}/{len(records)}")


def add_adherence(records: list[dict[str, Any]]) -> None:
    """The judge's score minus the base model's for the same prompt and seed."""
    base = {(r["prompt"], r["seed"]): r for r in records if r["run"] == "base"}
    for r in records:
        b = base.get((r["prompt"], r["seed"]))
        r["adherence"] = (r["judge"] - b["judge"]) if (b and r["judge"] is not None) else None


def aggregate_rows(
    records: list[dict[str, Any]],
    checks: dict[str, list[str]],
    image_clip: np.ndarray,
    floor: float,
    ceiling: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[(r["run"], r["step"])].append(i)

    def mean_of(idx: list[int], key: str, prompt: str | None = None) -> float | None:
        vals = [
            records[i][key]
            for i in idx
            if records[i].get(key) is not None and (prompt is None or records[i]["prompt"] == prompt)
        ]
        return float(np.mean(vals)) if vals else None

    def diversity(idx: list[int]) -> float | None:
        by_prompt = defaultdict(list)
        for i in idx:
            by_prompt[records[i]["prompt"]].append(i)
        vals = [
            1.0 - float(np.mean([image_clip[a] @ image_clip[b] for a, b in itertools.combinations(m, 2)]))
            for m in by_prompt.values()
            if len(m) > 1
        ]
        return float(np.mean(vals)) if vals else None

    rows = []
    for (run, step), idx in groups.items():
        likes = [records[i]["likeness"] for i in idx if records[i]["likeness"] is not None]
        like = float(np.mean(likes)) if likes else None
        rows.append(
            {
                "run": run,
                "step": step,
                "n": len(idx),
                "face_rate": sum(records[i]["face_found"] for i in idx) / len(idx),
                "likeness": like,
                "likeness_sem": float(np.std(likes, ddof=1) / np.sqrt(len(likes))) if len(likes) > 1 else None,
                "likeness_norm": (like - floor) / (ceiling - floor) if like is not None and ceiling != floor else None,
                "judge": mean_of(idx, "judge"),
                "adherence": mean_of(idx, "adherence"),
                "diversity": diversity(idx),
                "memorization": mean_of(idx, "memorization"),
                "per_prompt": {
                    pid: {
                        "likeness": mean_of(idx, "likeness", pid),
                        "judge": mean_of(idx, "judge", pid),
                        "adherence": mean_of(idx, "adherence", pid),
                    }
                    for pid in checks
                },
            }
        )
    rows.sort(key=lambda r: (r["run"] != "base", r["run"], step_order(r["step"])))
    return rows


def fmt(v: float | None, f: str) -> str:
    return f.format(v) if v is not None else "   -  "


def print_table(rows: list[dict[str, Any]], floor: float, ceiling: float, train_pairwise: float) -> None:
    print(
        f"\nlikeness floor (base) {floor:.4f}  ceiling (real photos) {ceiling:.4f}  "
        f"training pairwise CLIP {train_pairwise:.4f}\n"
    )
    print("        run  step  face%  likeness   norm  judge  adhere  divers   memo")
    for row in rows:
        print(
            f"{row['run']:>11} {row['step']:>5} {row['face_rate'] * 100:5.0f}% {fmt(row['likeness'], '{:9.4f}')} "
            f"{fmt(row['likeness_norm'], '{:6.2f}')} {fmt(row['judge'], '{:6.3f}')} "
            f"{fmt(row['adherence'], '{:+7.3f}')} {fmt(row['diversity'], '{:7.4f}')} "
            f"{fmt(row['memorization'], '{:6.4f}')}"
        )
