#!/usr/bin/env python3
"""Score a benchmark image set produced by qwen_2_1_bench_generate.py.

Runs in ComfyUI's venv (opencv, transformers >= Qwen3-VL, optional insightface); it imports
nothing from maltrainer. Every model it uses is already on this machine:

  * CLIP ViT-H/14 vision tower   ComfyUI/models/clip_vision/clip_vision_h.safetensors
  * Qwen3-VL-4B-Instruct          Hugging Face cache (judge for prompt adherence)
  * ArcFace (optional)            insightface model dir, if one is present

Metrics, per image, then aggregated per (run, step):

  likeness     cosine similarity between the generated face and the centroid of the training
               faces. ArcFace when available (a real identity embedding); otherwise CLIP-H on a
               Haar-cascade face crop -- a weaker proxy that also reacts to hair and lighting.
               Calibrated by two reference points computed identically:
                 floor   = the base model (no LoRA) on the same prompts and seeds
                 ceiling = each real training face vs the centroid of the *other* faces
               likeness_norm = (score - floor) / (ceiling - floor): 0 = generic, 1 = as close
               as another real photo of her.
  adherence    mean P("Yes") from Qwen3-VL-4B over each prompt's yes/no checks, minus the base
               model's value for the same prompt and seed. Negative means the LoRA is overriding
               what the prompt asked for.
  diversity    1 - mean pairwise CLIP-H similarity across seeds of one (run, step, prompt).
               Falling diversity is the compositional-collapse signal.
  memorization max CLIP-H similarity to any single training image; approaching the training
               images' own pairwise similarity means copying rather than generalizing.

Writes <bench>/scores.json (per image) and <bench>/summary.json (aggregates + references).

Usage:
    C:\\Development\\ComfyUI\\venv\\Scripts\\python.exe scripts\\qwen_2_1_bench_score.py --bench samples\\bench
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_2_1_bench_score_aggregate import (
    add_adherence,
    aggregate_rows,
    judge_records,
    load_training_images,
    print_table,
    score_images,
    step_order,
    train_reference,
)
from qwen_2_1_bench_score_models import (  # noqa: F401  (re-exported)
    CLIP_H,
    CLIP_MEAN,
    CLIP_STD,
    JUDGE_REPO,
    PROCESSOR_DIR,
    ClipVisionH,
    FaceEmbedder,
    YesNoJudge,
    free,
    l2,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--dataset", default=None, help="defaults to the path in manifest.json")
    parser.add_argument("--arcface-root", default=r"C:\Development\ComfyUI\models\insightface")
    parser.add_argument("--arcface-name", default="buffalo_l")
    parser.add_argument("--no-judge", action="store_true", help="skip the VLM adherence judge")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    bench = Path(args.bench)
    manifest = json.loads((bench / "manifest.json").read_text())
    checks = {p["id"]: p.get("checks", []) for p in manifest["prompts"]}
    records = manifest["images"]
    images = [Image.open(bench / r["path"]).convert("RGB") for r in records]
    train_images = load_training_images(Path(args.dataset or manifest["dataset"]))

    # ---- CLIP-H + faces ----------------------------------------------------------------
    clip = ClipVisionH(args.device)
    faces = FaceEmbedder(clip, args.arcface_root, args.arcface_name)
    print(f"face backend: {faces.backend}")

    reference = train_reference(clip, faces, train_images)
    image_clip = clip(images)
    score_images(records, image_clip, faces(images), reference)
    face_backend = faces.backend
    del clip, faces
    free()

    # ---- VLM adherence judge ------------------------------------------------------------
    if not args.no_judge:
        judge = YesNoJudge(args.device)
        judge_records(judge, records, images, checks)
        del judge
        free()
        add_adherence(records)

    # ---- aggregate ------------------------------------------------------------------------
    base_likes = [r["likeness"] for r in records if r["run"] == "base" and r["likeness"] is not None]
    floor = float(np.mean(base_likes)) if base_likes else 0.0
    ceiling = reference["ceiling"]
    rows = aggregate_rows(records, checks, image_clip, floor, ceiling)
    rows.sort(key=lambda r: (r["run"] != "base", r["run"], step_order(r["step"])))

    summary = {
        "face_backend": face_backend,
        "reference": {
            "likeness_floor_base": floor,
            "likeness_ceiling_real": ceiling,
            "training_faces": reference["faces"],
            "training_pairwise_clip": reference["pairwise"],
        },
        "rows": rows,
    }
    (bench / "scores.json").write_text(json.dumps(records, indent=1))
    (bench / "summary.json").write_text(json.dumps(summary, indent=2))
    print_table(rows, floor, ceiling, reference["pairwise"])


if __name__ == "__main__":
    main()
