#!/usr/bin/env python3
"""End-to-end sanity probe for the Qwen-Image-2.1 training stack.

Runs the *frozen base model* over the dataset exactly the way the trainer does -- same VAE
latents, same cached text embeddings, same flow-matching target -- and reports its MSE per
sigma band against two baselines:

  baseline_zero   loss of predicting 0. For unit-variance latents E||noise - x0||^2 ~= 2.0.
  baseline_noise  loss of predicting the noise alone (ignoring x0), i.e. a model that has
                  learned the noise schedule but nothing about the image.

A healthy stack scores far below both, and the gap widens as sigma falls (a nearly-clean
latent plus the right caption makes the velocity nearly determined). A score near
baseline_zero at every sigma means the conditioning, the latents, or the timestep
convention are wrong -- train nothing until this passes.

This is the check that would have caught the missing VAE quant_conv / stage shortcuts:
before that fix the base model scored at baseline across the board.

Usage:
    venv\\Scripts\\python.exe scripts\\qwen_2_1_stack_probe.py --config config\\train_qwen_image_2_1_feliciaday_adamw.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maltrainer.data.qwen_dataset import QwenImage21Dataset
from maltrainer.models.qwen_image_2_1.model import (
    apply_flow_shift,
    calculate_shift,
    load_qwen_image_2_1_transformer,
)
from maltrainer.models.qwen_image_2_1.text_encoder import load_qwen_image_2_1_text_encoder
from maltrainer.models.qwen_image_2_1.vae import load_qwen_image_2_1_vae
from maltrainer.utils.config import load_config

SIGMAS = (0.15, 0.3, 0.5, 0.7, 0.9)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=8, help="images to probe")
    parser.add_argument("--repeats", type=int, default=4, help="noise draws per image per sigma")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda")
    dtype = getattr(torch, config.get("dtype", "bfloat16"))
    model_config = config["model"]
    dataset_config = config["dataset"]

    dataset = QwenImage21Dataset(
        data_root=dataset_config["path"],
        resolution=dataset_config.get("resolution", 512),
        trigger_token=dataset_config.get("trigger"),
        default_caption=dataset_config.get("default_caption", ""),
        repeats=1,
    )
    items = [dataset[i] for i in range(min(args.samples, len(dataset)))]
    print(f"probing {len(items)} images at {dataset.height}x{dataset.width}")

    vae = load_qwen_image_2_1_vae(model_config["vae_path"], device=device, dtype=dtype)
    with torch.no_grad():
        latents = [
            vae.encode(item["pixel_values"].unsqueeze(0).to(device, dtype), deterministic=True).cpu() for item in items
        ]
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    stacked = torch.cat(latents).float()
    print(f"latent std {stacked.std():.4f} (expect ~1.0-1.15; ~0.8 means quant_conv is missing)")

    text_encoder = load_qwen_image_2_1_text_encoder(
        model_config["text_encoder_path"],
        asset_dir=model_config.get("tokenizer_path"),
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        embeds = [text_encoder.encode_prompts([item["caption"]], device=device).cpu() for item in items]
        # a deliberately wrong caption: the gap against it is what proves the text stream is read
        shuffled = embeds[1:] + embeds[:1]
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    transformer = load_qwen_image_2_1_transformer(model_config["transformer_path"], device=device, dtype=dtype)

    print(f"\n{'sigma':>6} {'matched':>9} {'mismatched':>11} {'zero':>8} {'noise-only':>11}")
    generator = torch.Generator(device=device).manual_seed(0)
    for sigma in SIGMAS:
        totals = {"matched": 0.0, "mismatched": 0.0, "zero": 0.0, "noise": 0.0}
        count = 0
        for latent, embed, wrong in tqdm(list(zip(latents, embeds, shuffled)), desc=f"sigma {sigma:.2f}", leave=False):
            x0 = latent.to(device, dtype)
            b, c, h, w = x0.shape
            mu = calculate_shift(h * w)
            shifted = apply_flow_shift(torch.full((b,), sigma, device=device, dtype=torch.float32), mu=mu)
            for _ in range(args.repeats):
                noise = torch.randn(x0.shape, device=device, dtype=dtype, generator=generator)
                s = shifted.view(b, 1, 1, 1).to(dtype)
                noisy = ((1.0 - s) * x0 + s * noise).permute(0, 2, 3, 1).reshape(b, h * w, c)
                target = (noise - x0).permute(0, 2, 3, 1).reshape(b, h * w, c).float()
                noise_flat = noise.permute(0, 2, 3, 1).reshape(b, h * w, c).float()
                with torch.no_grad():
                    for name, ctx in (("matched", embed), ("mismatched", wrong)):
                        pred = transformer(
                            hidden_states=noisy,
                            encoder_hidden_states=ctx.to(device, dtype),
                            timestep=shifted.to(dtype),
                            latent_h=h,
                            latent_w=w,
                        ).float()
                        totals[name] += torch.nn.functional.mse_loss(pred, target).item()
                totals["zero"] += target.pow(2).mean().item()
                totals["noise"] += torch.nn.functional.mse_loss(noise_flat, target).item()
                count += 1
        print(
            f"{sigma:>6.2f} {totals['matched'] / count:>9.4f} {totals['mismatched'] / count:>11.4f} "
            f"{totals['zero'] / count:>8.4f} {totals['noise'] / count:>11.4f}"
        )

    print(
        "\nPASS if `matched` sits well under `zero`/`noise-only` and drops as sigma falls,\n"
        "and if `matched` < `mismatched` (the model is actually reading the caption)."
    )


if __name__ == "__main__":
    main()
