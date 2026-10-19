#!/usr/bin/env python3
"""Sample every LoRA snapshot of a run from one model load, for picking a checkpoint by eye.

Reloading the text encoder and transformer per checkpoint costs about a minute each; here
they load once. The adapter is attached once too, and each checkpoint is swapped in by
copying weights into the already-attached layers -- every snapshot of a run shares the same
rank and module list, so the shapes always match. Setting the scale to 0 reproduces the base
model exactly, which gives a free no-LoRA reference in the same grid.

Writes one PNG per (checkpoint, seed) plus a contact sheet with checkpoints as rows and
seeds as columns.

Usage:
    venv\\Scripts\\python.exe scripts\\qwen_2_1_sweep_checkpoints.py \\
        --config config\\train_qwen_image_2_1_feliciaday_1500.yaml \\
        --prompt "feliciaday a photo of feliciaday, studio portrait, dark background" \\
        --outdir samples\\sweep --seeds 1 2
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRANetwork
from maltrainer.models.qwen_image_2_1.model import calculate_shift, load_qwen_image_2_1_transformer
from maltrainer.models.qwen_image_2_1.text_encoder import load_qwen_image_2_1_text_encoder
from maltrainer.models.qwen_image_2_1.vae import load_qwen_image_2_1_vae
from maltrainer.utils.config import load_config

VAE_SCALE_FACTOR = 16
SHIFT_TERMINAL = 0.02


def build_sigmas(num_steps: int, image_seq_len: int, device: torch.device) -> torch.Tensor:
    sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float32, device=device)
    shift = math.exp(calculate_shift(image_seq_len))
    sigmas = (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)
    one_minus = 1.0 - sigmas
    sigmas = 1.0 - one_minus / (one_minus[-1] / (1.0 - SHIFT_TERMINAL))
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def sort_key(path: Path) -> tuple[int, int]:
    """`..._step_600` -> 600; the unsuffixed final checkpoint sorts last."""
    match = re.search(r"_step_(\d+)$", path.stem)
    return (0, int(match.group(1))) if match else (1, 0)


def label_of(path: Path) -> str:
    match = re.search(r"_step_(\d+)$", path.stem)
    return f"step {match.group(1)}" if match else "final"


def swap_adapter(network: QwenImage21LoRANetwork, path: Path, scale: float) -> None:
    """Copy one checkpoint's weights into the adapter layers already attached to `network`.

    Every snapshot of a run shares rank and module list, so the attached layers never need
    rebuilding. `scale` multiplies alpha/rank; 0 reproduces the base model exactly.
    """
    from safetensors.torch import load_file

    layers = network.assistant_layers
    state = load_file(str(path))
    for key in state:
        if not key.endswith(".lora_A.weight"):
            continue
        module_path = key[len(network.prefix) : -len(".lora_A.weight")]
        layer: Any = layers[module_path.replace(".", "_")]
        layer.lora_down.weight.data.copy_(state[key].to(layer.lora_down.weight.dtype))
        up = state[key.replace(".lora_A.", ".lora_B.")]
        layer.lora_up.weight.data.copy_(up.to(layer.lora_up.weight.dtype))
        alpha = state.get(key.replace(".lora_A.weight", ".alpha"))
        rank = state[key].shape[0]
        layer.scale = (float(alpha.item()) if alpha is not None else float(rank)) / rank * scale


def sample_latents(
    transformer: Any, embeds: Any, sigmas: Any, seed: int, latent_h: int, latent_w: int, dtype: Any, device: Any
) -> torch.Tensor:
    """Euler flow-matching sample from pure noise; returns normalized latents [1, 64, h, w]."""
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((1, 64, latent_h, latent_w), device=device, dtype=torch.float32, generator=generator)
    for i in range(len(sigmas) - 1):
        packed = latents.to(dtype).permute(0, 2, 3, 1).reshape(1, latent_h * latent_w, 64)
        with torch.no_grad():
            velocity = transformer(
                hidden_states=packed,
                encoder_hidden_states=embeds,
                timestep=sigmas[i].reshape(1).to(dtype),
                latent_h=latent_h,
                latent_w=latent_w,
            ).float()
        velocity = velocity.reshape(1, latent_h, latent_w, 64).permute(0, 3, 1, 2)
        latents = latents + (sigmas[i + 1] - sigmas[i]) * velocity
    return latents


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--no-base", action="store_true", help="skip the scale-0 reference row")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    device = torch.device("cuda")
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(config["output"]["path"])
    checkpoints = sorted(run_dir.glob("*.safetensors"), key=sort_key)
    if not checkpoints:
        raise SystemExit(f"no checkpoints in {run_dir}")
    print(f"{len(checkpoints)} checkpoints: {', '.join(label_of(c) for c in checkpoints)}")

    edge = args.size // (VAE_SCALE_FACTOR * 2) * (VAE_SCALE_FACTOR * 2)
    latent_h = latent_w = edge // VAE_SCALE_FACTOR

    text_encoder = load_qwen_image_2_1_text_encoder(
        model_config["text_encoder_path"], asset_dir=model_config.get("tokenizer_path"), device=device
    )
    with torch.no_grad():
        embeds = text_encoder.encode_prompts([args.prompt], device=device).to(device, dtype)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    transformer: Any = load_qwen_image_2_1_transformer(model_config["transformer_path"], device=device, dtype=dtype)
    network: Any = QwenImage21LoRANetwork(transformer, target_modules=["__none__"])
    network.load_assistant_adapter(checkpoints[0], multiplier=1.0, device=device, dtype=dtype)
    print(f"adapter attached: {len(network.assistant_layers)} modules")

    def load_checkpoint(path: Path, scale: float) -> None:
        swap_adapter(network, path, scale)

    sigmas = build_sigmas(args.steps, latent_h * latent_w, device)

    def sample(seed: int) -> torch.Tensor:
        return sample_latents(transformer, embeds, sigmas, seed, latent_h, latent_w, dtype, device)

    rows: list[tuple[str, list[torch.Tensor]]] = []
    if not args.no_base:
        load_checkpoint(checkpoints[0], scale=0.0)  # scale 0 => base model
        rows.append(("base", [sample(s) for s in args.seeds]))
        print("  base (no LoRA) done")
    for path in checkpoints:
        load_checkpoint(path, scale=args.lora_scale)
        rows.append((label_of(path), [sample(s) for s in args.seeds]))
        print(f"  {label_of(path)} done")

    # Not `del`: load_checkpoint and sample close over both names.
    transformer = network = None
    gc.collect()
    torch.cuda.empty_cache()

    vae = load_qwen_image_2_1_vae(model_config["vae_path"], device=device, dtype=dtype, include_decoder=True)
    grid: list[tuple[str, list[Image.Image]]] = []
    for label, latents_list in rows:
        images = []
        for seed, latents in zip(args.seeds, latents_list, strict=False):
            with torch.no_grad():
                decoded = vae.decode(latents.to(dtype))
            array = decoded[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
            image = Image.fromarray(array.round().astype("uint8"))
            image.save(outdir / f"{label.replace(' ', '_')}_seed{seed}.png")
            images.append(image)
        grid.append((label, images))

    pad, header = 4, 22
    cell = grid[0][1][0].width
    sheet = Image.new("RGB", (len(args.seeds) * (cell + pad) + pad, len(grid) * (cell + pad + header) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (label, images) in enumerate(grid):
        y = pad + r * (cell + pad + header)
        draw.text((pad, y + 5), f"{label}   (seeds {', '.join(map(str, args.seeds))})", fill="black")
        for c, image in enumerate(images):
            sheet.paste(image, (pad + c * (cell + pad), y + header))
    sheet_path = outdir / "contact_sheet.png"
    sheet.save(sheet_path)
    print(f"\nwrote {len(grid) * len(args.seeds)} images + {sheet_path}")


if __name__ == "__main__":
    main()
