"""Scoring models for scripts/qwen_2_1_bench_score.py: CLIP-H vision tower, face embedder, Qwen3-VL judge."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

CLIP_H = r"C:\Development\ComfyUI\models\clip_vision\clip_vision_h.safetensors"
JUDGE_REPO = "Qwen/Qwen3-VL-4B-Instruct"
# the snapshot carries weights but no processor; Qwen3-VL's processor is size-independent
PROCESSOR_DIR = Path(__file__).resolve().parents[1] / "maltrainer" / "models" / "minimax_h3" / "assets" / "text_encoder"
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def l2(v: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12))


def free() -> None:
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
        self.model = self.model.to(device, torch.float16).eval()  # type: ignore[arg-type]  # @wraps stub types `self` as arg 1
        self.device = device

    def _prep(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        scale = 224 / min(image.size)
        image = image.resize(
            (max(224, round(image.width * scale)), max(224, round(image.height * scale))), Image.Resampling.BICUBIC
        )
        left, top = (image.width - 224) // 2, (image.height - 224) // 2
        array = np.asarray(image.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255.0
        return cast(np.ndarray, ((array - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1))

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
            out: list[np.ndarray | None] = []
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
            crops.append(image.convert("RGB").crop(box).resize((224, 224), Image.Resampling.BICUBIC))
            index.append(i)
        out = [None] * len(images)
        if crops:
            for i, emb in zip(index, self.clip(crops), strict=False):
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
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()  # type: ignore[arg-type]  # @wraps stub types `self` as arg 1
        tokenizer = AutoTokenizer.from_pretrained(path)
        self.processor = cast(Any, Qwen3VLProcessor)(
            image_processor=cast(Any, AutoImageProcessor).from_pretrained(PROCESSOR_DIR),
            video_processor=cast(Any, AutoVideoProcessor).from_pretrained(PROCESSOR_DIR),
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
