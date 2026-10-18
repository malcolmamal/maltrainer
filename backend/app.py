from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Set

import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


DATASETS_ENV = "DATASETS_PATH"
COMFYUI_MODELS_ENV = "COMFYUI_MODELS_PATH"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PROGRESS_RE = re.compile(r"(?:Step:\s*)?(\d+)\s*/\s*(\d+)")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("backend")


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    label: str
    default_lr: float
    note: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """One trainable model family: which script runs it, which template seeds its config."""

    id: str
    label: str
    script: str
    template: str
    optimizers: tuple[OptimizerSpec, ...]
    default_optimizer: str
    default_duration_mode: str  # "epochs" | "steps"
    default_epochs: int
    default_max_steps: int
    # True when the trainer honours train.max_steps itself. Otherwise a step budget is
    # rounded up to whole epochs before the config is written.
    native_max_steps: bool
    # Required checkpoint files, found anywhere under COMFYUI_MODELS_PATH/<subfolder>/.
    model_files: tuple[tuple[str, str, str], ...] = ()  # (config key, subfolder, file name)
    # Measured seconds per step (RTX 5090, 512px, rank 16) for the UI's time estimate; 0 = unknown.
    sec_per_step: float = 0.0
    setup_seconds: float = 0.0
    # Legacy runs write straight to output/<dataset>; newer models get one folder per run.
    per_run_output: bool = False
    # Auto step budget, used when the user gives no step count: aim for target_epochs epochs of
    # this dataset, rounded to 50 and kept inside [min_auto_steps, max_auto_steps] -- the range
    # the benchmark actually covered. target_epochs 0 = no auto budget.
    target_epochs: int = 0
    min_auto_steps: int = 0
    max_auto_steps: int = 0
    # Guidance derived from the benchmark (0 / False = off).
    epoch_bounds: tuple[int, int] = (0, 0)  # warn below the first / above the second
    max_benchmarked_steps: int = 0
    # Training is square-only for now: non-square images are squashed, so the plan warns about them.
    square_only: bool = False
    resolution_note: str = ""
    # A whole recipe per optimizer, where the optimizer changes more than its learning rate.
    optimizer_templates: tuple[tuple[str, str], ...] = ()  # (optimizer, template path)
    # Output file prefix (the name the LoRA pickers parse) and whether a custom trigger also
    # rewrites the default caption to "a photo of <trigger>".
    output_prefix: str = ""
    caption_from_trigger: bool = False


MODELS: Dict[str, ModelSpec] = {
    spec.id: spec
    for spec in (
        ModelSpec(
            id="zimage",
            label="Z-Image (Base)",
            script="train_zimage_lora.py",
            template="templates/zimage_base_template.yaml",
            optimizers=(
                OptimizerSpec("adamw8bit", "AdamW 8-bit", 1e-4),
                OptimizerSpec("prodigy", "Prodigy", 1.0, "learning rate is Prodigy's multiplier"),
            ),
            default_optimizer="adamw8bit",
            default_duration_mode="epochs",
            default_epochs=100,
            default_max_steps=2000,
            native_max_steps=False,
        ),
        ModelSpec(
            id="qwen_image_2_1",
            label="Qwen-Image 2.1",
            script="train_qwen_image_2_1_lora.py",
            template="templates/qwen_image_2_1_template.yaml",
            optimizers=(
                OptimizerSpec("automagic3", "Automagic3 (recommended)", 1e-6,
                              "self-tuning; the learning rate is only its starting point"),
                OptimizerSpec("adamw8bit", "AdamW 8-bit", 1e-4, "keep checkpoints 1250-1750"),
                OptimizerSpec("prodigy", "Prodigy", 1.0,
                              "not recommended on Qwen: overcooks past ~1500 steps"),
            ),
            default_optimizer="automagic3",
            default_duration_mode="steps",
            default_epochs=70,
            default_max_steps=1500,
            native_max_steps=True,
            model_files=(
                ("transformer_path", "diffusion_models", "qwen_image_2.1_int8_convrot.safetensors"),
                ("vae_path", "vae", "qwen_image_2.1_vae_bf16.safetensors"),
                ("text_encoder_path", "text_encoders", "qwen3vl_8b_int8_convrot.safetensors"),
            ),
            sec_per_step=0.62,
            setup_seconds=20.0,
            per_run_output=True,
            target_epochs=70,
            min_auto_steps=1000,
            max_auto_steps=2500,
            epoch_bounds=(20, 120),
            max_benchmarked_steps=2500,
            square_only=True,
            resolution_note=(
                "Training resolution is square 512×512 for now. More resolutions and "
                "aspect-ratio bucketing are planned."
            ),
            output_prefix="qwen21_",
            caption_from_trigger=True,
        ),
        ModelSpec(
            id="krea2",
            label="Krea 2",
            script="train_krea2_lora.py",
            template="templates/krea2_template.yaml",
            optimizers=(
                OptimizerSpec("automagic3", "Automagic3 (recommended)", 1e-6,
                              "self-tuning; per module family, 16-step sign window"),
                OptimizerSpec("prodigy", "Prodigy (OneTrainer recipe)", 1.0,
                              "rank 16 / alpha 1, 200 warmup steps, unshifted logit-normal"),
                OptimizerSpec("adamw8bit", "AdamW 8-bit (Fizgig Standard)", 1e-4,
                              "rank 32, every Linear, flat 1e-4, EMA 0.98"),
            ),
            default_optimizer="automagic3",
            default_duration_mode="steps",
            default_epochs=70,
            default_max_steps=1500,
            native_max_steps=True,
            model_files=(
                ("transformer_path", "diffusion_models", "krea2_raw_fp8_scaled.safetensors"),
                ("vae_path", "vae", "qwen_image_vae.safetensors"),
                ("text_encoder_path", "text_encoders", "qwen3vl_4b_fp8_scaled.safetensors"),
            ),
            per_run_output=True,
            resolution_note="Aspect-ratio buckets around 512×512 on a 16 px grid.",
            optimizer_templates=(("prodigy", "templates/krea2_prodigy_template.yaml"),
                                 ("adamw8bit", "templates/krea2_adamw_template.yaml")),
            output_prefix="krea2_",
            caption_from_trigger=True,
        ),
        ModelSpec(
            id="flux3_klein",
            label="FLUX.3 [klein] (Base)",
            script="train_flux3_klein_lora.py",
            template="templates/flux3_klein_template.yaml",
            optimizers=(
                OptimizerSpec("automagic3", "Automagic3 (recommended)", 1e-6,
                              "self-tuning; per module family, 16-step sign window"),
                OptimizerSpec("adamw8bit", "AdamW 8-bit", 1e-4, "the diffusers FLUX.3 example's rate"),
                OptimizerSpec("prodigy", "Prodigy", 1.0, "learning rate is Prodigy's multiplier"),
            ),
            default_optimizer="automagic3",
            default_duration_mode="steps",
            default_epochs=70,
            default_max_steps=1500,
            native_max_steps=True,
            model_files=(
                ("transformer_path", "diffusion_models", "flux3_image_klein_int8_convrot.safetensors"),
                ("vae_path", "vae", "flux2_vae.safetensors"),
                ("text_encoder_path", "text_encoders", "qwen3vl_4b_int8_convrot.safetensors"),
            ),
            per_run_output=True,
            resolution_note="Aspect-ratio buckets around 512×512 on a 16 px grid.",
            output_prefix="flux3_klein_",
            caption_from_trigger=True,
        ),
        ModelSpec(
            id="moss_image",
            label="MOSS-Image (Base)",
            script="train_moss_image_lora.py",
            template="templates/moss_image_automagic_template.yaml",
            optimizers=(
                OptimizerSpec("automagic3", "Automagic3 (recommended)", 1e-6,
                              "self-tuning; per module family, 8-step sign window"),
                OptimizerSpec("prodigy", "Prodigy", 1.0,
                              "learning rate is Prodigy's multiplier; 100 warmup steps, no EMA"),
                OptimizerSpec("adamw8bit", "AdamW 8-bit", 1e-4, "flat rate, weight decay 0.01"),
            ),
            default_optimizer="automagic3",
            default_duration_mode="steps",
            default_epochs=70,
            default_max_steps=1500,
            native_max_steps=True,
            model_files=(
                ("transformer_path", "diffusion_models", "moss_image_base_bf16.safetensors"),
                ("vae_path", "vae", "flux2_vae.safetensors"),
                ("text_encoder_path", "text_encoders", "qwen3.5_9b_int8_convrot.safetensors"),
            ),
            per_run_output=True,
            resolution_note="Aspect-ratio buckets around 512×512 on a 16 px grid.",
            optimizer_templates=(("prodigy", "templates/moss_image_prodigy_template.yaml"),
                                 ("adamw8bit", "templates/moss_image_adamw_template.yaml")),
            output_prefix="moss_image_",
            caption_from_trigger=True,
        ),
    )
}


class DatasetInfo(BaseModel):
    name: str
    path: str
    image_count: int


class OptimizerInfo(BaseModel):
    name: str
    label: str
    default_lr: float
    note: str


class ModelInfo(BaseModel):
    id: str
    label: str
    optimizers: list[OptimizerInfo]
    default_optimizer: str
    default_duration_mode: str
    default_epochs: int
    default_max_steps: int
    native_max_steps: bool
    sec_per_step: float
    models_root: Optional[str]
    missing_files: list[str]
    auto_steps: bool
    target_epochs: int
    min_auto_steps: int
    max_auto_steps: int
    resolution_note: str


class StartTrainRequest(BaseModel):
    dataset: str = Field(..., min_length=1)
    model: str = "zimage"
    optimizer: Optional[str] = None
    learning_rate: Optional[float] = Field(None, gt=0)
    duration_mode: Optional[str] = Field(None, pattern="^(epochs|steps)$")
    epochs: Optional[int] = Field(None, ge=1, le=100000)
    max_steps: Optional[int] = Field(None, ge=1, le=10000000)
    save_every: Optional[int] = Field(None, ge=1)
    trigger: Optional[str] = None
    dry_run: bool = False
    dry_run_step_time: float = 0.025


class TrainPlan(BaseModel):
    model: str
    dataset: str
    optimizer: str
    learning_rate: float
    duration_mode: str
    image_count: int
    steps_per_epoch: int
    epochs: int
    total_steps: int
    save_every: int
    output_path: str
    duration: str
    auto_steps: bool
    estimated_minutes: Optional[float]
    notes: list[str]
    warnings: list[str]


class StartTrainResponse(BaseModel):
    job_id: str
    dataset: str
    output_path: str
    config_path: str
    command: list[str]


class JobStatusResponse(BaseModel):
    job_id: str
    dataset: str
    output_path: str
    config_path: str
    status: str
    started_at: float
    ended_at: Optional[float]
    exit_code: Optional[int]
    progress: Dict[str, Any]
    model: str = "zimage"
    optimizer: str = ""
    duration: str = ""


class JobListResponse(BaseModel):
    items: list[JobStatusResponse]


@dataclass
class TrainingJob:
    job_id: str
    dataset: str
    output_path: str
    config_path: str
    command: list[str]
    process: Optional[asyncio.subprocess.Process]
    started_at: float
    dry_run: bool
    dry_run_step_time: float
    model: str = "zimage"
    optimizer: str = ""
    duration: str = ""
    status: str = "running"
    ended_at: Optional[float] = None
    exit_code: Optional[int] = None
    progress: Dict[str, Any] = field(default_factory=lambda: {"step": 0, "total": 0})
    websockets: Set[WebSocket] = field(default_factory=set)
    log_buffer: Deque[dict] = field(default_factory=lambda: deque(maxlen=2000))
    reader_task: Optional[asyncio.Task] = None

    async def start_readers(self) -> None:
        if not self.process:
            return
        async def _emit_log(line: str, stream_name: str, replace: bool) -> None:
            text = line.rstrip("\n")
            if text == "":
                return
            match = PROGRESS_RE.search(text)
            is_progress = bool(match)
            payload = {
                "type": "log",
                "stream": stream_name,
                "line": text,
                "replace": replace,
                "is_progress": is_progress,
            }
            await self.broadcast(payload)
            if match:
                step = int(match.group(1))
                total = int(match.group(2))
                self.progress.update({"step": step, "total": total})
                await self.broadcast({"type": "progress", "step": step, "total": total})

        async def _read_stream(stream: asyncio.StreamReader, stream_name: str) -> None:
            buffer = ""
            while True:
                chunk = await stream.read(1024)
                if not chunk:
                    if buffer:
                        await _emit_log(buffer, stream_name, replace=False)
                    break
                buffer += chunk.decode(errors="ignore")

                while True:
                    nl = buffer.find("\n")
                    cr = buffer.find("\r")
                    if nl == -1 and cr == -1:
                        break

                    if nl == -1:
                        sep_index = cr
                        sep_char = "\r"
                    elif cr == -1:
                        sep_index = nl
                        sep_char = "\n"
                    else:
                        sep_index = min(nl, cr)
                        sep_char = "\r" if sep_index == cr else "\n"

                    line = buffer[:sep_index]
                    buffer = buffer[sep_index + 1 :]
                    await _emit_log(line, stream_name, replace=sep_char == "\r")

        stdout = self.process.stdout
        stderr = self.process.stderr
        if stdout is None or stderr is None:
            return
        await asyncio.gather(_read_stream(stdout, "stdout"), _read_stream(stderr, "stderr"))

    async def monitor(self) -> None:
        if not self.process:
            return
        self.reader_task = asyncio.create_task(self.start_readers())
        await self.process.wait()
        self.exit_code = self.process.returncode
        self.ended_at = time.time()
        self.status = "exited" if self.exit_code == 0 else "failed"
        logger.info("Job finished: %s (%s) status=%s", self.job_id, self.dataset, self.status)
        await self.broadcast(
            {
                "type": "status",
                "status": self.status,
                "exit_code": self.exit_code,
            }
        )
        await launch_next_job()

    async def broadcast(self, payload: dict) -> None:
        self.log_buffer.append(payload)
        stale: list[WebSocket] = []
        for ws in self.websockets:
            try:
                await ws.send_json(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.websockets.discard(ws)


app = FastAPI(title="Maltrainer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, TrainingJob] = {}
queued_job_ids: Deque[str] = deque()
job_lock = asyncio.Lock()


def datasets_root() -> Path:
    root = os.environ.get(DATASETS_ENV, "datasets")
    return Path(root).resolve()


def output_root() -> Path:
    return Path("output").resolve()


def comfyui_models_root() -> Optional[Path]:
    root = os.environ.get(COMFYUI_MODELS_ENV)
    return Path(root).resolve() if root else None


_model_file_cache: Dict[str, tuple[float, Dict[str, str], list[str]]] = {}
MODEL_FILE_CACHE_SECONDS = 30.0


def resolve_model_files(spec: ModelSpec) -> tuple[Dict[str, str], list[str]]:
    """Find each required checkpoint under COMFYUI_MODELS_PATH/<subfolder>/ (any depth).

    Returns (config key -> absolute path, missing "subfolder/file" entries). Cached briefly
    because the plan endpoint is called on every form change and model folders can be large.
    """
    root = comfyui_models_root()
    cache_key = f"{spec.id}|{root}"
    cached = _model_file_cache.get(cache_key)
    if cached and time.time() - cached[0] < MODEL_FILE_CACHE_SECONDS:
        return dict(cached[1]), list(cached[2])

    found: Dict[str, str] = {}
    missing: list[str] = []
    for key, subfolder, name in spec.model_files:
        path: Optional[Path] = None
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


def load_template(spec: ModelSpec, dataset_name: str, optimizer: Optional[str] = None) -> Dict[str, Any]:
    template_path = Path(dict(spec.optimizer_templates).get(optimizer or "", spec.template))
    if not template_path.exists():
        raise HTTPException(status_code=500, detail=f"Template not found: {spec.template}")
    template_text = template_path.read_text(encoding="utf-8")
    template_text = template_text.replace("[person_to_train]", dataset_name)
    root = comfyui_models_root()
    if root is not None:
        template_text = template_text.replace("[comfyui_models]", root.as_posix())
    return yaml.safe_load(template_text)


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for root, _, files in os.walk(path):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTS:
                count += 1
    return count


NON_SQUARE_TOLERANCE = 0.03  # |w - h| / max(w, h) above this counts as non-square
_scan_cache: Dict[str, tuple[float, tuple[int, int, Optional[tuple[int, int]]]]] = {}
SCAN_CACHE_SECONDS = 30.0


def scan_images(path: Path) -> tuple[int, int, Optional[tuple[int, int]]]:
    """(image count, non-square count, one non-square example size). Reads image headers only.

    Cached briefly because the plan endpoint runs on every form change.
    """
    key = str(path)
    cached = _scan_cache.get(key)
    if cached and time.time() - cached[0] < SCAN_CACHE_SECONDS:
        return cached[1]

    from PIL import Image

    count, non_square, example = 0, 0, None
    for root, _, files in os.walk(path):
        for fname in files:
            if Path(fname).suffix.lower() not in SUPPORTED_EXTS:
                continue
            count += 1
            try:
                with Image.open(Path(root) / fname) as img:
                    width, height = img.size
            except Exception:
                continue
            if abs(width - height) / max(width, height, 1) > NON_SQUARE_TOLERANCE:
                non_square += 1
                example = example or (width, height)
    result = (count, non_square, example)
    _scan_cache[key] = (time.time(), result)
    return result


def get_model_spec(model_id: str) -> ModelSpec:
    spec = MODELS.get(model_id)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_id}'. Available: {', '.join(MODELS)}")
    return spec


def build_plan(request: StartTrainRequest) -> tuple[TrainPlan, Dict[str, Any], ModelSpec, list[str]]:
    """Turn a UI request into a concrete config. Returns (plan, config, spec, missing model files)."""
    spec = get_model_spec(request.model)
    dataset_path = datasets_root() / request.dataset
    if not dataset_path.is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found")

    config = load_template(spec, request.dataset, request.optimizer or spec.default_optimizer)
    dataset_cfg = config.setdefault("dataset", {})
    train_cfg = config.setdefault("train", {})
    output_cfg = config.setdefault("output", {})
    warnings: list[str] = []

    missing: list[str] = []
    if spec.model_files:
        found, missing = resolve_model_files(spec)
        config.setdefault("model", {}).update(found)
        if missing:
            where = comfyui_models_root() or f"(set {COMFYUI_MODELS_ENV})"
            warnings.append(f"Model files not found under {where}: {', '.join(missing)}")

    optimizers = {o.name: o for o in spec.optimizers}
    optimizer = request.optimizer or spec.default_optimizer
    if optimizer not in optimizers:
        raise HTTPException(
            status_code=400,
            detail=f"{spec.label} does not support optimizer '{optimizer}'. Use one of: {', '.join(optimizers)}",
        )
    learning_rate = request.learning_rate or optimizers[optimizer].default_lr
    train_cfg["optimizer"] = optimizer
    train_cfg["learning_rate"] = learning_rate
    if spec.id == "qwen_image_2_1" and optimizer == "prodigy":
        warnings.append("Prodigy overcooks Qwen-Image 2.1 LoRAs in benchmarks; keep checkpoints at or below ~1500 steps.")

    trigger = (request.trigger or "").strip()
    if trigger:
        dataset_cfg["trigger"] = trigger
        if spec.caption_from_trigger:
            dataset_cfg["default_caption"] = f"a photo of {trigger}"

    notes: list[str] = []
    if spec.square_only:
        image_count, non_square, example = scan_images(dataset_path)
        if non_square:
            resolution = int(dataset_cfg.get("resolution", 512) or 512)
            warnings.append(
                f"{non_square} of {image_count} images are not square (e.g. {example[0]}×{example[1]}). "
                f"Training is square {resolution}×{resolution} only for now, so they will be squashed to "
                "fit. Crop them to square first; more resolutions and aspect-ratio bucketing are planned."
            )
    else:
        image_count = count_images(dataset_path)
    if image_count == 0:
        warnings.append("The dataset has no images.")
    repeats = max(int(dataset_cfg.get("repeats", 1) or 1), 1)
    batch_size = max(int(train_cfg.get("batch_size", 1) or 1), 1)
    grad_accum = max(int(train_cfg.get("gradient_accumulation_steps", 1) or 1), 1)
    batches = math.ceil(max(image_count, 1) * repeats / batch_size)
    steps_per_epoch = max(math.ceil(batches / grad_accum), 1)

    mode = request.duration_mode or spec.default_duration_mode
    auto_steps = False
    if mode == "steps":
        max_steps = request.max_steps
        if not max_steps and spec.target_epochs and image_count > 0:
            wanted = spec.target_epochs * steps_per_epoch
            rounded = int(round(wanted / 50.0)) * 50
            max_steps = min(max(rounded, spec.min_auto_steps), spec.max_auto_steps)
            auto_steps = True
            note = (
                f"Steps chosen automatically: {image_count} images × ~{spec.target_epochs} epochs "
                f"= {wanted} steps"
            )
            if rounded != wanted:
                note += f", rounded to {rounded}"
            if max_steps != rounded:
                note += (
                    f", then kept within {spec.min_auto_steps}–{spec.max_auto_steps} steps, the range the "
                    f"benchmark covered, giving {max_steps}"
                )
            notes.append(note + ".")
        max_steps = max_steps or spec.default_max_steps
        epochs = math.ceil(max_steps / steps_per_epoch)
        train_cfg["num_epochs"] = epochs
        if spec.native_max_steps:
            train_cfg["max_steps"] = max_steps
            total_steps = max_steps
            duration = f"{max_steps} steps"
        else:
            train_cfg.pop("max_steps", None)
            total_steps = epochs * steps_per_epoch
            duration = f"{epochs} epochs ({total_steps} steps)"
            if total_steps != max_steps:
                warnings.append(
                    f"{spec.label} trains whole epochs: {max_steps} steps rounded up to "
                    f"{epochs} epochs = {total_steps} steps."
                )
        run_label = f"{max_steps}steps"
    else:
        epochs = request.epochs or spec.default_epochs
        train_cfg["num_epochs"] = epochs
        if spec.native_max_steps:
            train_cfg["max_steps"] = 0
        else:
            train_cfg.pop("max_steps", None)
        total_steps = epochs * steps_per_epoch
        duration = f"{epochs} epochs ({total_steps} steps)"
        run_label = f"{epochs}ep"

    save_every = request.save_every or int(train_cfg.get("save_every", 500))
    train_cfg["save_every"] = save_every

    if image_count > 0 and spec.epoch_bounds != (0, 0):
        low, high = spec.epoch_bounds
        effective_epochs = total_steps / steps_per_epoch
        if effective_epochs < low:
            warnings.append(
                f"Only {effective_epochs:.1f} epochs. In the benchmark, likeness was at roughly 70% of its "
                f"plateau after 11 epochs and 90–95% after 22, so runs under ~{low} epochs risk being "
                "under-trained. Raise max steps if you can."
            )
        elif effective_epochs > high:
            warnings.append(
                f"{effective_epochs:.1f} epochs is beyond what was benchmarked (up to {high - 1}); "
                "very small datasets can over-fit or memorize."
            )
    if spec.max_benchmarked_steps and total_steps > spec.max_benchmarked_steps:
        warnings.append(
            f"{total_steps} steps is longer than any benchmarked run ({spec.max_benchmarked_steps} at most). "
            "Automagic3 stayed stable up to there; beyond it is untested."
        )

    if spec.per_run_output:
        base = output_root() / spec.id / f"{request.dataset}-{optimizer}-{run_label}"
        output_dir, n = base, 2
        while output_dir.exists():
            output_dir = base.with_name(f"{base.name}-{n}")
            n += 1
        output_cfg["name"] = f"{spec.output_prefix}{output_dir.name}"
    else:
        output_dir = output_root() / request.dataset
    output_cfg["path"] = str(output_dir)
    dataset_cfg["path"] = str(dataset_path)

    estimated = None
    if spec.sec_per_step and not request.dry_run:
        estimated = round((total_steps * spec.sec_per_step + spec.setup_seconds) / 60.0, 1)

    plan = TrainPlan(
        model=spec.id,
        dataset=request.dataset,
        optimizer=optimizer,
        learning_rate=learning_rate,
        duration_mode=mode,
        image_count=image_count,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        total_steps=total_steps,
        save_every=save_every,
        output_path=str(output_dir),
        duration=duration,
        auto_steps=auto_steps,
        estimated_minutes=estimated,
        notes=notes,
        warnings=warnings,
    )
    return plan, config, spec, missing


def job_to_status(job: "TrainingJob") -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        dataset=job.dataset,
        output_path=job.output_path,
        config_path=job.config_path,
        status=job.status,
        started_at=job.started_at,
        ended_at=job.ended_at,
        exit_code=job.exit_code,
        progress=job.progress,
        model=job.model,
        optimizer=job.optimizer,
        duration=job.duration,
    )


async def spawn_process(command: list[str]) -> asyncio.subprocess.Process:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return await asyncio.create_subprocess_exec(
        *command,
        cwd=str(Path(".").resolve()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )


async def launch_job(job: TrainingJob) -> None:
    job.status = "running"
    job.started_at = time.time()
    job.process = await spawn_process(job.command)
    logger.info("Job started: %s (%s)", job.job_id, job.dataset)
    asyncio.create_task(job.monitor())


async def launch_next_job() -> None:
    async with job_lock:
        running = any(j.status == "running" for j in jobs.values())
        if running:
            return
        while queued_job_ids:
            next_id = queued_job_ids.popleft()
            job = jobs.get(next_id)
            if job and job.status == "queued":
                await launch_job(job)
                return


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/datasets", response_model=list[DatasetInfo])
async def list_datasets() -> list[DatasetInfo]:
    root = datasets_root()
    if not root.exists():
        return []
    items: list[DatasetInfo] = []
    for entry in root.iterdir():
        if entry.is_dir():
            items.append(
                DatasetInfo(
                    name=entry.name,
                    path=str(entry),
                    image_count=count_images(entry),
                )
            )
    return sorted(items, key=lambda d: d.name.lower())


@app.get("/api/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    items = []
    for spec in MODELS.values():
        _, missing = resolve_model_files(spec) if spec.model_files else ({}, [])
        root = comfyui_models_root()
        items.append(
            ModelInfo(
                id=spec.id,
                label=spec.label,
                optimizers=[OptimizerInfo(name=o.name, label=o.label, default_lr=o.default_lr, note=o.note)
                            for o in spec.optimizers],
                default_optimizer=spec.default_optimizer,
                default_duration_mode=spec.default_duration_mode,
                default_epochs=spec.default_epochs,
                default_max_steps=spec.default_max_steps,
                native_max_steps=spec.native_max_steps,
                sec_per_step=spec.sec_per_step,
                models_root=str(root) if (root and spec.model_files) else None,
                missing_files=missing,
                auto_steps=bool(spec.target_epochs),
                target_epochs=spec.target_epochs,
                min_auto_steps=spec.min_auto_steps,
                max_auto_steps=spec.max_auto_steps,
                resolution_note=spec.resolution_note,
            )
        )
    return items


@app.post("/api/plan", response_model=TrainPlan)
async def plan_training(request: StartTrainRequest) -> TrainPlan:
    """Preview what /api/train would do with this request, without starting anything."""
    plan, _, _, _ = build_plan(request)
    return plan


@app.post("/api/train", response_model=StartTrainResponse)
async def start_training(request: StartTrainRequest) -> StartTrainResponse:
    plan, config, spec, missing = build_plan(request)
    if missing and not request.dry_run:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot train {spec.label}: missing model files {', '.join(missing)}. "
                f"Set {COMFYUI_MODELS_ENV} to your ComfyUI models folder "
                f"(currently {comfyui_models_root() or 'unset'}) and restart the backend."
            ),
        )

    output_dir = Path(plan.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    total_steps = plan.total_steps

    script_path = Path(spec.script).resolve()
    python_exe = sys.executable
    cmd = [python_exe, "-u", str(script_path), "--config", str(config_path)]
    if request.dry_run:
        cmd.append("--dry-run")
        if request.dry_run_step_time is not None:
            cmd.extend(["--dry-run-step-time", str(request.dry_run_step_time)])

    job_id = uuid.uuid4().hex
    job = TrainingJob(
        job_id=job_id,
        dataset=request.dataset,
        output_path=str(output_dir),
        config_path=str(config_path),
        command=cmd,
        process=None,
        started_at=time.time(),
        dry_run=request.dry_run,
        dry_run_step_time=request.dry_run_step_time,
        model=spec.id,
        optimizer=plan.optimizer,
        duration=plan.duration + (" · dry run" if request.dry_run else ""),
    )
    job.progress.update({"step": 0, "total": total_steps})
    # Not "running" until launched - otherwise the check below finds this job itself and
    # every submission queues, even on an idle trainer.
    job.status = "queued"
    jobs[job_id] = job
    logger.info("Job added: %s (%s)", job.job_id, job.dataset)

    async with job_lock:
        running = any(j.status == "running" for j in jobs.values() if j is not job)
        if running:
            job.status = "queued"
            queued_job_ids.append(job_id)
            logger.info("Job queued: %s (%s)", job.job_id, job.dataset)
        else:
            await launch_job(job)

    return StartTrainResponse(
        job_id=job_id,
        dataset=request.dataset,
        output_path=str(output_dir),
        config_path=str(config_path),
        command=cmd,
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_status(job)


@app.get("/api/jobs", response_model=JobListResponse)
async def job_list() -> JobListResponse:
    items = [job_to_status(job) for job in jobs.values()]
    items.sort(key=lambda j: j.started_at, reverse=True)
    return JobListResponse(items=items)


@app.post("/api/jobs/{job_id}/start")
async def start_specific_job(job_id: str) -> Dict[str, str]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    async with job_lock:
        running = any(j.status == "running" for j in jobs.values())
        if running:
            raise HTTPException(status_code=409, detail="Another job is running")
        if job.status != "queued":
            raise HTTPException(status_code=400, detail="Job is not queued")
        queued_job_ids.remove(job_id) if job_id in queued_job_ids else None
        await launch_job(job)
    return {"status": "ok"}


@app.post("/api/jobs/start-next")
async def start_next_job() -> Dict[str, str]:
    await launch_next_job()
    return {"status": "ok"}


@app.websocket("/ws/jobs/{job_id}")
async def job_ws(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    job = jobs.get(job_id)
    if not job:
        await ws.send_json({"type": "error", "message": "Job not found"})
        await ws.close()
        return

    job.websockets.add(ws)
    try:
        await ws.send_json(
            {
                "type": "job",
                "job": {
                    "job_id": job.job_id,
                    "dataset": job.dataset,
                    "output_path": job.output_path,
                    "config_path": job.config_path,
                    "status": job.status,
                    "started_at": job.started_at,
                    "progress": job.progress,
                    "model": job.model,
                    "optimizer": job.optimizer,
                    "duration": job.duration,
                },
            }
        )
        for payload in job.log_buffer:
            await ws.send_json(payload)

        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        job.websockets.discard(ws)
    except Exception:
        job.websockets.discard(ws)
        await ws.close()
