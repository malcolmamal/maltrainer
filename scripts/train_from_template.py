#!/usr/bin/env python3
"""Fill a maltrainer template for one dataset and run its trainer -- the web UI's path, from a shell.

    python scripts/train_from_template.py krea2 feliciaday
    python scripts/train_from_template.py krea2 feliciaday --optimizer prodigy --max-steps 2000
    python scripts/train_from_template.py krea2 feliciaday --dry-run

Placeholders: [person_to_train] -> the dataset name, [comfyui_models] -> COMFYUI_MODELS_PATH.
Model files are looked up anywhere under <models>/<subfolder>/ like the web UI does, the dataset
under DATASETS_PATH (default ./datasets). The filled config is written next to the outputs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

# model id -> (trainer script, default template, {optimizer: template}, {config key: (subfolder, file)})
MODELS = {
    "krea2": (
        "train_krea2_lora.py",
        "templates/krea2_template.yaml",
        {"prodigy": "templates/krea2_prodigy_template.yaml", "adamw8bit": "templates/krea2_adamw_template.yaml"},
        {
            "transformer_path": ("diffusion_models", "krea2_raw_fp8_scaled.safetensors"),
            "vae_path": ("vae", "qwen_image_vae.safetensors"),
            "text_encoder_path": ("text_encoders", "qwen3vl_4b_fp8_scaled.safetensors"),
        },
    ),
    "flux3_klein": (
        "train_flux3_klein_lora.py",
        "templates/flux3_klein_template.yaml",
        {},
        {
            "transformer_path": ("diffusion_models", "flux3_image_klein_int8_convrot.safetensors"),
            "vae_path": ("vae", "flux2_vae.safetensors"),
            "text_encoder_path": ("text_encoders", "qwen3vl_4b_int8_convrot.safetensors"),
        },
    ),
    "moss_image": (
        "train_moss_image_lora.py",
        "templates/moss_image_automagic_template.yaml",
        {
            "prodigy": "templates/moss_image_prodigy_template.yaml",
            "adamw8bit": "templates/moss_image_adamw_template.yaml",
        },
        {
            "transformer_path": ("diffusion_models", "moss_image_base_bf16.safetensors"),
            "vae_path": ("vae", "flux2_vae.safetensors"),
            "text_encoder_path": ("text_encoders", "qwen3.5_9b_int8_convrot.safetensors"),
        },
    ),
    "qwen_image_2_1": (
        "train_qwen_image_2_1_lora.py",
        "templates/qwen_image_2_1_template.yaml",
        {},
        {
            "transformer_path": ("diffusion_models", "qwen_image_2.1_int8_convrot.safetensors"),
            "vae_path": ("vae", "qwen_image_2.1_vae_bf16.safetensors"),
            "text_encoder_path": ("text_encoders", "qwen3vl_8b_int8_convrot.safetensors"),
        },
    ),
}


def find_model(root: Path, subfolder: str, name: str) -> Path | None:
    base = root / subfolder
    if (base / name).is_file():
        return base / name
    return next((p for p in base.rglob(name) if p.is_file()), None) if base.is_dir() else None


def _place_models(
    args: argparse.Namespace, config: dict[str, Any], files: dict[str, tuple[str, str]], text: str
) -> int | None:
    """Point the config at the ComfyUI model files. A message and exit code 1 when they cannot be found."""
    if args.comfyui_models:
        missing = []
        for key, (subfolder, name) in files.items():
            found = find_model(Path(args.comfyui_models), subfolder, name)
            if found:
                config["model"][key] = str(found)
            else:
                missing.append(f"{subfolder}/{name}")
        if missing and not args.dry_run:
            print(f"Not found under {args.comfyui_models}: {', '.join(missing)}", file=sys.stderr)
            return 1
    elif "[comfyui_models]" in text and not args.dry_run:
        print("Set COMFYUI_MODELS_PATH (or --comfyui-models) to your ComfyUI models folder.", file=sys.stderr)
        return 1
    return None


def _apply_overrides(args: argparse.Namespace, config: dict[str, Any]) -> None:
    train = config.setdefault("train", {})
    if args.optimizer:
        train["optimizer"] = args.optimizer
    if args.learning_rate:
        train["learning_rate"] = args.learning_rate
    if args.epochs:
        train["num_epochs"], train["max_steps"] = args.epochs, 0
    if args.max_steps:
        train["max_steps"] = args.max_steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("dataset", help="dataset folder name under DATASETS_PATH")
    parser.add_argument("--optimizer", help="e.g. automagic3 (default), prodigy, adamw8bit")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-steps", type=int, help="> 0 trains exactly this many steps")
    parser.add_argument("--epochs", type=int, help="train whole epochs instead of max_steps")
    parser.add_argument("--trigger", help="trigger word (default: the dataset name)")
    parser.add_argument("--comfyui-models", default=os.environ.get("COMFYUI_MODELS_PATH"))
    parser.add_argument("--datasets", default=os.environ.get("DATASETS_PATH", str(ROOT / "datasets")))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script, template, optimizer_templates, files = MODELS[args.model]
    template_path = ROOT / optimizer_templates.get(args.optimizer or "", template)
    text = template_path.read_text(encoding="utf-8").replace("[person_to_train]", args.dataset)
    if args.comfyui_models:
        text = text.replace("[comfyui_models]", Path(args.comfyui_models).as_posix())
    config = yaml.safe_load(text)

    dataset_path = (Path(args.datasets) / args.dataset).resolve()  # the trainer runs from ROOT, not from here
    if not dataset_path.is_dir():
        print(f"Dataset folder not found: {dataset_path}", file=sys.stderr)
        return 1
    config["dataset"]["path"] = str(dataset_path)
    if args.trigger:
        config["dataset"]["trigger"] = args.trigger
        config["dataset"]["default_caption"] = f"a photo of {args.trigger}"

    code = _place_models(args, config, files, text)
    if code is not None:
        return code

    _apply_overrides(args, config)

    output = Path(config["output"]["path"])
    output.mkdir(parents=True, exist_ok=True)
    rendered = output / "config.yaml"
    rendered.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Config written to {rendered}")

    # absolute: the trainer runs from ROOT, so a path relative to where this was launched would not be found
    command = [sys.executable, str(ROOT / script), "--config", str(rendered.resolve())] + (
        ["--dry-run"] if args.dry_run else []
    )
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
