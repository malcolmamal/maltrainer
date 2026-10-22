"""Where the backend looks for things: the datasets, the output and the ComfyUI models folders (from the
environment), the checkpoints a model needs, its config template, and a dataset's images."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, cast

import yaml
from fastapi import HTTPException

from backend.specs import ModelSpec
from maltrainer.data.walk import count_files, walk_visible

DATASETS_ENV = "DATASETS_PATH"
COMFYUI_MODELS_ENV = "COMFYUI_MODELS_PATH"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def datasets_root() -> Path:
    root = os.environ.get(DATASETS_ENV, "datasets")
    return Path(root).resolve()


def output_root() -> Path:
    return Path("output").resolve()


def comfyui_models_root() -> Path | None:
    root = os.environ.get(COMFYUI_MODELS_ENV)
    return Path(root).resolve() if root else None


_model_file_cache: dict[str, tuple[float, dict[str, str], list[str]]] = {}
MODEL_FILE_CACHE_SECONDS = 30.0


def resolve_model_files(spec: ModelSpec) -> tuple[dict[str, str], list[str]]:
    """Find each required checkpoint under COMFYUI_MODELS_PATH/<subfolder>/ (any depth).

    Returns (config key -> absolute path, missing "subfolder/file" entries). Cached briefly
    because the plan endpoint is called on every form change and model folders can be large.
    """
    root = comfyui_models_root()
    cache_key = f"{spec.id}|{root}"
    cached = _model_file_cache.get(cache_key)
    if cached and time.time() - cached[0] < MODEL_FILE_CACHE_SECONDS:
        return dict(cached[1]), list(cached[2])

    found: dict[str, str] = {}
    missing: list[str] = []
    for key, subfolder, name in spec.model_files:
        path: Path | None = None
        if root is not None:
            base = root / subfolder
            if (base / name).is_file():
                path = base / name
            elif base.is_dir():
                path = next((p for p in base.rglob(name) if p.is_file()), None)
        if path is not None:
            found[key] = str(path)
        else:
            missing.append(f"{subfolder}/{name}")
    _model_file_cache[cache_key] = (time.time(), found, missing)
    return dict(found), list(missing)


def load_template(spec: ModelSpec, dataset_name: str, optimizer: str | None = None) -> dict[str, Any]:
    template_path = Path(dict(spec.optimizer_templates).get(optimizer or "", spec.template))
    if not template_path.exists():
        raise HTTPException(status_code=500, detail=f"Template not found: {template_path.as_posix()}")
    template_text = template_path.read_text(encoding="utf-8")
    template_text = template_text.replace("[person_to_train]", dataset_name)
    root = comfyui_models_root()
    if root is not None:
        template_text = template_text.replace("[comfyui_models]", root.as_posix())
    return cast(dict[str, Any], yaml.safe_load(template_text))


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return count_files(path, SUPPORTED_EXTS)


NON_SQUARE_TOLERANCE = 0.03  # |w - h| / max(w, h) above this counts as non-square
_scan_cache: dict[str, tuple[float, tuple[int, int, tuple[int, int] | None]]] = {}
SCAN_CACHE_SECONDS = 30.0


def scan_images(path: Path) -> tuple[int, int, tuple[int, int] | None]:
    """(image count, non-square count, one non-square example size). Reads image headers only.

    Cached briefly because the plan endpoint runs on every form change.
    """
    key = str(path)
    cached = _scan_cache.get(key)
    if cached and time.time() - cached[0] < SCAN_CACHE_SECONDS:
        return cached[1]

    from PIL import Image

    count, non_square, example = 0, 0, None
    for root, _, files in walk_visible(path):
        for fname in files:
            if Path(fname).suffix.lower() not in SUPPORTED_EXTS:
                continue
            count += 1
            try:
                with Image.open(Path(root) / fname) as img:
                    width, height = img.size
            except (OSError, ValueError, Image.DecompressionBombError):
                continue
            if abs(width - height) / max(width, height, 1) > NON_SQUARE_TOLERANCE:
                non_square += 1
                example = example or (width, height)
    result = (count, non_square, example)
    _scan_cache[key] = (time.time(), result)
    return result
