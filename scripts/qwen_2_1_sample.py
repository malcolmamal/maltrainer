#!/usr/bin/env python3
"""Minimal flow-matching sampler for Qwen-Image-2.1, with optional LoRA.

Exists so a training run can be judged from inside maltrainer without a round trip through
ComfyUI. The sigma schedule mirrors the reference pipeline: linspace(1, 1/N, N), exponential
dynamic shift from the token count, then stretched so the final sigma lands on
`shift_terminal`. Qwen-Image-2.1 is guidance-distilled, so guidance_scale 1.0 (no CFG) is the
intended way to sample it.

Components load and unload one at a time -- text encoder, then transformer, then VAE decoder
-- so peak VRAM is one model, not three.

Usage:
    venv\\Scripts\\python.exe scripts\\qwen_2_1_sample.py --config config\\train_qwen_image_2_1_feliciaday_adamw.yaml \\
        --prompt "feliciaday a photo of a woman, studio portrait" --out out.png
    ... --lora output\\qwen21_feliciaday_short\\qwen21_feliciaday_short_step_500.safetensors
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRANetwork
from maltrainer.models.qwen_image_2_1.model import calculate_shift, load_qwen_image_2_1_transformer
from maltrainer.models.qwen_image_2_1.text_encoder import load_qwen_image_2_1_text_encoder
from maltrainer.models.qwen_image_2_1.vae import load_qwen_image_2_1_vae
from maltrainer.utils.config import load_config

VAE_SCALE_FACTOR = 16
SHIFT_TERMINAL = 0.02


def build_sigmas(num_steps: int, image_seq_len: int, device: torch.device) -> torch.Tensor:
    """Reference schedule: linear sigmas, exponential dynamic shift, terminal stretch, then 0."""
    sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float32, device=device)
    shift = math.exp(calculate_shift(image_seq_len))
    sigmas = (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)
    one_minus = 1.0 - sigmas
    sigmas = 1.0 - one_minus / (one_minus[-1] / (1.0 - SHIFT_TERMINAL))
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lora", default=None, help="LoRA safetensors to apply")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--size", type=int, default=512, help="square output edge in pixels")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    device = torch.device("cuda")
    dtype = getattr(torch, config.get("dtype", "bfloat16"))

    edge = args.size // (VAE_SCALE_FACTOR * 2) * (VAE_SCALE_FACTOR * 2)  # 32px grid
    latent_h = latent_w = edge // VAE_SCALE_FACTOR
    print(f"sampling {edge}x{edge} ({latent_h * latent_w} image tokens), {args.steps} steps")

    text_encoder = load_qwen_image_2_1_text_encoder(
        model_config["text_encoder_path"],
        asset_dir=model_config.get("tokenizer_path"),
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        embeds = text_encoder.encode_prompts([args.prompt], device=device).to(device, dtype)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    transformer = load_qwen_image_2_1_transformer(
        model_config["transformer_path"], device=device, dtype=dtype
    )
    if args.lora:
        # target_modules that match nothing: inject no fresh adapters, then attach the
        # trained ones through the same frozen-adapter path the trainer uses
        network = QwenImage21LoRANetwork(transformer, target_modules=["__none__"])
        attached = network.load_assistant_adapter(
            args.lora, multiplier=args.lora_scale, device=device, dtype=dtype
        )
        print(f"applied LoRA {Path(args.lora).name}: {attached} modules at scale {args.lora_scale}")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    latents = torch.randn(
        (1, 64, latent_h, latent_w), device=device, dtype=torch.float32, generator=generator
    )
    sigmas = build_sigmas(args.steps, latent_h * latent_w, device)

    for i in range(args.steps):
        sigma = sigmas[i]
        packed = latents.to(dtype).permute(0, 2, 3, 1).reshape(1, latent_h * latent_w, 64)
        with torch.no_grad():
            velocity = transformer(
                hidden_states=packed,
                encoder_hidden_states=embeds,
                timestep=sigma.reshape(1).to(dtype),
                latent_h=latent_h,
                latent_w=latent_w,
            ).float()
        velocity = velocity.reshape(1, latent_h, latent_w, 64).permute(0, 3, 1, 2)
        latents = latents + (sigmas[i + 1] - sigma) * velocity

    del transformer
    gc.collect()
    torch.cuda.empty_cache()

    vae = load_qwen_image_2_1_vae(
        model_config["vae_path"], device=device, dtype=dtype, include_decoder=True
    )
    with torch.no_grad():
        image = vae.decode(latents.to(dtype))
    array = (image[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
