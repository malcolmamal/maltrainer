"""Turn a UI request into a concrete trainer config and a plan the UI can show: what will run, for how long,
and what to watch out for."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from backend import files
from backend.guidance import run_length_warnings, steps_for_dataset
from backend.registry import get_model_spec
from backend.schemas import StartTrainRequest, TrainPlan
from backend.specs import ModelSpec


@dataclass
class Duration:
    mode: str
    epochs: int
    total_steps: int
    text: str  # "<n> steps" / "<n> epochs (<m> steps)"
    run_label: str  # in the output folder's name
    auto_steps: bool = False


def build_plan(request: StartTrainRequest) -> tuple[TrainPlan, dict[str, Any], ModelSpec, list[str]]:
    """Turn a UI request into a concrete config. Returns (plan, config, spec, missing model files)."""
    spec = get_model_spec(request.model)
    dataset_path = files.datasets_root() / request.dataset
    if not dataset_path.is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found")

    config = files.load_template(spec, request.dataset, request.optimizer or spec.default_optimizer)
    dataset_cfg = config.setdefault("dataset", {})
    train_cfg = config.setdefault("train", {})
    output_cfg = config.setdefault("output", {})
    warnings: list[str] = []
    notes: list[str] = []

    missing = _model_files(spec, config, warnings)
    optimizer, learning_rate = _optimizer(spec, request, train_cfg, warnings)
    trigger = (request.trigger or "").strip()
    if trigger:
        dataset_cfg["trigger"] = trigger
        if spec.caption_from_trigger:
            dataset_cfg["default_caption"] = f"a photo of {trigger}"

    image_count = _image_count(spec, dataset_path, dataset_cfg, warnings)
    repeats = max(int(dataset_cfg.get("repeats", 1) or 1), 1)
    batch_size = max(int(train_cfg.get("batch_size", 1) or 1), 1)
    grad_accum = max(int(train_cfg.get("gradient_accumulation_steps", 1) or 1), 1)
    batches = math.ceil(max(image_count, 1) * repeats / batch_size)
    steps_per_epoch = max(math.ceil(batches / grad_accum), 1)

    duration = _duration(request, spec, train_cfg, (image_count, steps_per_epoch), (warnings, notes))
    save_every = request.save_every or int(train_cfg.get("save_every", 500))
    train_cfg["save_every"] = save_every
    effective_epochs = duration.total_steps / steps_per_epoch if image_count > 0 else None
    run_length_warnings(spec, effective_epochs, duration.total_steps, warnings)

    output_dir = _output_dir(spec, request, optimizer, duration.run_label, output_cfg)
    output_cfg["path"] = str(output_dir)
    dataset_cfg["path"] = str(dataset_path)

    estimated = None
    if spec.sec_per_step and not request.dry_run:
        estimated = round((duration.total_steps * spec.sec_per_step + spec.setup_seconds) / 60.0, 1)
    plan = TrainPlan(
        model=spec.id,
        dataset=request.dataset,
        optimizer=optimizer,
        learning_rate=learning_rate,
        duration_mode=duration.mode,
        image_count=image_count,
        steps_per_epoch=steps_per_epoch,
        epochs=duration.epochs,
        total_steps=duration.total_steps,
        save_every=save_every,
        output_path=str(output_dir),
        duration=duration.text,
        auto_steps=duration.auto_steps,
        estimated_minutes=estimated,
        notes=notes,
        warnings=warnings,
    )
    return plan, config, spec, missing


def _model_files(spec: ModelSpec, config: dict[str, Any], warnings: list[str]) -> list[str]:
    """The checkpoints found go into config.model; the ones missing are warned about and returned."""
    missing: list[str] = []
    if spec.model_files:
        found, missing = files.resolve_model_files(spec)
        config.setdefault("model", {}).update(found)
        if missing:
            where = files.comfyui_models_root() or f"(set {files.COMFYUI_MODELS_ENV})"
            warnings.append(f"Model files not found under {where}: {', '.join(missing)}")
    return missing


def _optimizer(
    spec: ModelSpec, request: StartTrainRequest, train_cfg: dict[str, Any], warnings: list[str]
) -> tuple[str, float]:
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
        warnings.append(
            "Prodigy overcooks Qwen-Image 2.1 LoRAs in benchmarks; keep checkpoints at or below ~1500 steps."
        )
    return optimizer, learning_rate


def _image_count(spec: ModelSpec, dataset_path: Path, dataset_cfg: dict[str, Any], warnings: list[str]) -> int:
    if spec.square_only:
        image_count, non_square, example = files.scan_images(dataset_path)
        if non_square and example is not None:
            resolution = int(dataset_cfg.get("resolution", 512) or 512)
            warnings.append(
                f"{non_square} of {image_count} images are not square (e.g. {example[0]}×{example[1]}). "
                f"Training is square {resolution}×{resolution} only for now, so they will be squashed to "
                "fit. Crop them to square first; more resolutions and aspect-ratio bucketing are planned."
            )
    else:
        image_count = files.count_images(dataset_path)
    if image_count == 0:
        warnings.append("The dataset has no images.")
    return image_count


def _duration(
    request: StartTrainRequest,
    spec: ModelSpec,
    train_cfg: dict[str, Any],
    dataset: tuple[int, int],
    out: tuple[list[str], list[str]],
) -> Duration:
    """The run's length (in steps or epochs, as asked) written into train.num_epochs / max_steps."""
    image_count, steps_per_epoch = dataset
    warnings, notes = out
    mode = request.duration_mode or spec.default_duration_mode
    if mode != "steps":
        epochs = request.epochs or spec.default_epochs
        train_cfg["num_epochs"] = epochs
        if spec.native_max_steps:
            train_cfg["max_steps"] = 0
        else:
            train_cfg.pop("max_steps", None)
        total_steps = epochs * steps_per_epoch
        return Duration(mode, epochs, total_steps, f"{epochs} epochs ({total_steps} steps)", f"{epochs}ep")

    max_steps = request.max_steps
    auto_steps = False
    if not max_steps and spec.target_epochs and image_count > 0:
        max_steps, note = steps_for_dataset(spec, image_count, steps_per_epoch)
        auto_steps = True
        notes.append(note)
    max_steps = max_steps or spec.default_max_steps
    epochs = math.ceil(max_steps / steps_per_epoch)
    train_cfg["num_epochs"] = epochs
    if spec.native_max_steps:
        train_cfg["max_steps"] = max_steps
        total_steps, text = max_steps, f"{max_steps} steps"
    else:
        train_cfg.pop("max_steps", None)
        total_steps = epochs * steps_per_epoch
        text = f"{epochs} epochs ({total_steps} steps)"
        if total_steps != max_steps:
            warnings.append(
                f"{spec.label} trains whole epochs: {max_steps} steps rounded up to "
                f"{epochs} epochs = {total_steps} steps."
            )
    return Duration(mode, epochs, total_steps, text, f"{max_steps}steps", auto_steps)


def _output_dir(
    spec: ModelSpec, request: StartTrainRequest, optimizer: str, run_label: str, output_cfg: dict[str, Any]
) -> Path:
    """New models get a folder per run (numbered when it exists); legacy ones write to output/<dataset>."""
    if not spec.per_run_output:
        return files.output_root() / request.dataset
    base = files.output_root() / spec.id / f"{request.dataset}-{optimizer}-{run_label}"
    output_dir, n = base, 2
    while output_dir.exists():
        output_dir = base.with_name(f"{base.name}-{n}")
        n += 1
    output_cfg["name"] = f"{spec.output_prefix}{output_dir.name}"
    return output_dir
