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
import gc
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")

CLIP_H = r"C:\Development\ComfyUI\models\clip_vision\clip_vision_h.safetensors"
JUDGE_REPO = "Qwen/Qwen3-VL-4B-Instruct"
# the snapshot carries weights but no processor; Qwen3-VL's processor is size-independent
PROCESSOR_DIR = Path(__file__).resolve().parents[1] / "maltrainer" / "models" / "minimax_h3" / "assets" / "text_encoder"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def free():
    gc.collect()
    torch.cuda.empty_cache()


class ClipVisionH:
    """OpenCLIP ViT-H/14 vision tower from ComfyUI's HF-format file. Image embeddings only."""

    def __init__(self, device: str):
        from safetensors.torch import load_file
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

        config = CLIPVisionConfig(
            hidden_size=1280,
            intermediate_size=5120,
            num_hidden_layers=32,
            num_attention_heads=16,
            patch_size=14,
            image_size=224,
            projection_dim=1024,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
        )
        self.model = CLIPVisionModelWithProjection(config)
        state = {k: v for k, v in load_file(CLIP_H).items() if not k.endswith("position_ids")}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [m for m in missing if not m.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"clip_vision_h mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.model = self.model.to(device, torch.float16).eval()
        self.device = device

    def _prep(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        scale = 224 / min(image.size)
        image = image.resize(
            (max(224, round(image.width * scale)), max(224, round(image.height * scale))), Image.BICUBIC
        )
        left, top = (image.width - 224) // 2, (image.height - 224) // 2
        array = np.asarray(image.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255.0
        return ((array - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(images), 32):
            batch = torch.from_numpy(np.stack([self._prep(im) for im in images[i : i + 32]]))
            with torch.no_grad():
                emb = self.model(pixel_values=batch.to(self.device, torch.float16)).image_embeds
            out.append(emb.float().cpu().numpy())
        return l2(np.concatenate(out))


class FaceEmbedder:
    """ArcFace when a model is present, else CLIP-H on a Haar-cascade face crop."""

    def __init__(self, clip: ClipVisionH, arcface_root: str | None, arcface_name: str):
        self.clip = clip
        self.app = None
        if arcface_root:
            model_dir = Path(arcface_root) / "models" / arcface_name
            if model_dir.is_dir() and any(model_dir.glob("*.onnx")):
                from insightface.app import FaceAnalysis

                self.app = FaceAnalysis(
                    name=arcface_name,
                    root=arcface_root,
                    allowed_modules=["detection", "recognition"],
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))
            else:
                print(f"no ArcFace model in {model_dir}; falling back to CLIP-H face crops")
        self.backend = "arcface" if self.app else "clip-h-facecrop"
        if not self.app:
            import cv2

            self.cv2 = cv2
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def __call__(self, images: list[Image.Image]) -> list[np.ndarray | None]:
        if self.app:
            out = []
            for image in images:
                faces = self.app.get(np.asarray(image.convert("RGB"))[:, :, ::-1])
                if not faces:
                    out.append(None)
                    continue
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                out.append(l2(face.normed_embedding.astype(np.float32)))
            return out

        crops, index = [], []
        for i, image in enumerate(images):
            rgb = np.asarray(image.convert("RGB"))
            gray = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)
            min_side = max(24, min(gray.shape) // 12)
            boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
            if len(boxes) == 0:
                continue
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
            cx, cy, half = x + w / 2, y + h / 2, max(w, h) * 0.65
            box = (
                int(max(0, cx - half)),
                int(max(0, cy - half)),
                int(min(rgb.shape[1], cx + half)),
                int(min(rgb.shape[0], cy + half)),
            )
            crops.append(image.convert("RGB").crop(box).resize((224, 224), Image.BICUBIC))
            index.append(i)
        out: list[np.ndarray | None] = [None] * len(images)
        if crops:
            for i, emb in zip(index, self.clip(crops)):
                out[i] = emb
        return out


class YesNoJudge:
    """VQAScore-style adherence: P(Yes) for a yes/no question, from Qwen3-VL-4B-Instruct."""

    TEMPLATE = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q} "
        "Answer with Yes or No.<|im_end|>\n<|im_start|>assistant\n"
    )

    def __init__(self, device: str):
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            AutoVideoProcessor,
            Qwen3VLForConditionalGeneration,
            Qwen3VLProcessor,
        )

        path = snapshot_download(JUDGE_REPO, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path)
        self.processor = Qwen3VLProcessor(
            image_processor=AutoImageProcessor.from_pretrained(PROCESSOR_DIR),
            video_processor=AutoVideoProcessor.from_pretrained(PROCESSOR_DIR),
            tokenizer=tokenizer,
        )
        self.yes = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ("Yes", " Yes", "yes")]
        self.no = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ("No", " No", "no")]
        self.device = device

    def __call__(self, image: Image.Image, question: str) -> float:
        inputs = self.processor(
            text=[self.TEMPLATE.format(q=question)], images=[image.convert("RGB")], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        yes = torch.logsumexp(logits[self.yes], 0)
        no = torch.logsumexp(logits[self.no], 0)
        return float(torch.sigmoid(yes - no))


def load_training_images(dataset: Path) -> list[Image.Image]:
    paths = sorted(p for p in dataset.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return [Image.open(p).convert("RGB") for p in paths]


def step_order(step: str) -> int:
    return 10**9 if step == "final" else (-1 if step == "base" else int(step))


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

    image_clip = clip(images)
    for r, emb, face in zip(records, image_clip, faces(images)):
        r["face_found"] = face is not None
        r["likeness"] = float(face @ centroid) if face is not None else None
        r["memorization"] = float((train_clip @ emb).max())
    face_backend = faces.backend
    del clip, faces
    free()

    # ---- VLM adherence judge ------------------------------------------------------------
    if not args.no_judge:
        judge = YesNoJudge(args.device)
        for i, (r, image) in enumerate(zip(records, images)):
            qs = checks.get(r["prompt"], [])
            r["checks"] = {q: judge(image, q) for q in qs}
            r["judge"] = float(np.mean(list(r["checks"].values()))) if qs else None
            if (i + 1) % 25 == 0:
                print(f"  judged {i + 1}/{len(records)}")
        del judge
        free()
        base = {(r["prompt"], r["seed"]): r for r in records if r["run"] == "base"}
        for r in records:
            b = base.get((r["prompt"], r["seed"]))
            r["adherence"] = (r["judge"] - b["judge"]) if (b and r["judge"] is not None) else None

    # ---- aggregate ------------------------------------------------------------------------
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[(r["run"], r["step"])].append(i)
    base_likes = [r["likeness"] for r in records if r["run"] == "base" and r["likeness"] is not None]
    floor = float(np.mean(base_likes)) if base_likes else 0.0

    def mean_of(idx, key, prompt=None):
        vals = [
            records[i][key]
            for i in idx
            if records[i].get(key) is not None and (prompt is None or records[i]["prompt"] == prompt)
        ]
        return float(np.mean(vals)) if vals else None

    def diversity(idx):
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

    summary = {
        "face_backend": face_backend,
        "reference": {
            "likeness_floor_base": floor,
            "likeness_ceiling_real": ceiling,
            "training_faces": int(len(train_faces)),
            "training_pairwise_clip": train_pairwise,
        },
        "rows": rows,
    }
    (bench / "scores.json").write_text(json.dumps(records, indent=1))
    (bench / "summary.json").write_text(json.dumps(summary, indent=2))

    def fmt(v, f):
        return f.format(v) if v is not None else "   -  "

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


if __name__ == "__main__":
    main()
