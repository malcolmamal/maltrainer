#!/usr/bin/env python3
"""Generate the image set for an optimizer / checkpoint benchmark.

For every run config given, samples every checkpoint under every (prompt, seed) pair, plus
one base-model (no LoRA) reference set shared by all runs. The text encoder, transformer and
VAE each load exactly once: prompts are encoded up front, the adapter is attached once and
each checkpoint is swapped in by weight copy, and all latents are decoded at the end.

All runs must share model paths and resolution -- the point is to vary only the optimizer.

Output layout (consumed by qwen_2_1_bench_score.py):
    <outdir>/manifest.json
    <outdir>/base/<prompt_id>_s<seed>.png
    <outdir>/<run>/<step>/<prompt_id>_s<seed>.png

Usage:
    venv\\Scripts\\python.exe scripts\\qwen_2_1_bench_generate.py --outdir samples\\bench \\
        --configs config\\train_qwen_image_2_1_feliciaday_bench_adamw.yaml \\
                  config\\train_qwen_image_2_1_feliciaday_bench_prodigy.yaml \\
                  config\\train_qwen_image_2_1_feliciaday_bench_automagic3.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRANetwork
from maltrainer.models.qwen_image_2_1.model import load_qwen_image_2_1_transformer
from maltrainer.models.qwen_image_2_1.text_encoder import load_qwen_image_2_1_text_encoder
from maltrainer.models.qwen_image_2_1.vae import load_qwen_image_2_1_vae
from maltrainer.utils.config import load_config
from qwen_2_1_sweep_adapter import sort_key, swap_adapter
from qwen_2_1_sweep_checkpoints import VAE_SCALE_FACTOR, build_sigmas, sample_latents

# `prompt` is what the model sees. `checks` are yes/no questions a VLM judge asks about the
# result to score prompt adherence -- phrased without the trigger word, which the judge has
# never heard of. Two of the three are deliberately far from the studio-portrait training set.
DEFAULT_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "portrait",
        "prompt": "feliciaday a photo of feliciaday, studio portrait, dark background",
        "checks": [
            "Is this a portrait photograph of a woman?",
            "Is the background dark?",
        ],
    },
    {
        "id": "knight",
        "prompt": "feliciaday as a medieval knight in silver plate armor, castle courtyard, cinematic photo",
        "checks": [
            "Is the person wearing metal plate armor?",
            "Is the setting a castle or castle courtyard?",
        ],
    },
    {
        "id": "cafe",
        "prompt": "feliciaday laughing at a table in a busy coffee shop, candid photo, warm window light, medium shot",
        "checks": [
            "Is the person laughing or smiling broadly?",
            "Is the setting the inside of a coffee shop or cafe?",
        ],
    },
]


def step_of(path: Path) -> str:
    match = re.search(r"_step_(\d+)$", path.stem)
    return match.group(1) if match else "final"


def run_name(config: dict[str, Any]) -> str:
    """bench_adamw -> adamw; falls back to the output folder name."""
    return Path(config["output"]["path"]).name.removeprefix("bench_")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--prompts-json", default=None, help="override the default prompt set")
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts_json).read_text()) if args.prompts_json else DEFAULT_PROMPTS
    configs = [load_config(c) for c in args.configs]
    model_config = configs[0]["model"]
    for c in configs[1:]:
        if c["model"] != model_config:
            raise SystemExit("all runs must share model paths")
    device = torch.device("cuda")
    dtype = getattr(torch, configs[0].get("dtype", "bfloat16"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = []
    for config in configs:
        checkpoints = sorted(Path(config["output"]["path"]).glob("*.safetensors"), key=sort_key)
        if not checkpoints:
            raise SystemExit(f"no checkpoints in {config['output']['path']}")
        runs.append((run_name(config), checkpoints))
        print(f"{run_name(config)}: {', '.join(step_of(c) for c in checkpoints)}")

    edge = args.size // (VAE_SCALE_FACTOR * 2) * (VAE_SCALE_FACTOR * 2)
    latent_h = latent_w = edge // VAE_SCALE_FACTOR
    total = (1 + sum(len(c) for _, c in runs)) * len(prompts) * len(args.seeds)
    print(f"{total} samples at {edge}px, {args.steps} steps")

    text_encoder = load_qwen_image_2_1_text_encoder(
        model_config["text_encoder_path"],
        asset_dir=model_config.get("tokenizer_path"),
        device=device,
    )
    with torch.no_grad():
        embeds = {p["id"]: text_encoder.encode_prompts([p["prompt"]], device=device).to(device, dtype) for p in prompts}
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()

    transformer = load_qwen_image_2_1_transformer(model_config["transformer_path"], device=device, dtype=dtype)
    network = QwenImage21LoRANetwork(transformer, target_modules=["__none__"])
    network.load_assistant_adapter(runs[0][1][0], multiplier=1.0, device=device, dtype=dtype)
    sigmas = build_sigmas(args.steps, latent_h * latent_w, device)

    jobs = [("base", "base", runs[0][1][0], 0.0)]
    jobs += [(name, step_of(ckpt), ckpt, 1.0) for name, ckpts in runs for ckpt in ckpts]

    latents: dict[tuple[str, str, str, int], torch.Tensor] = {}
    started, done = time.time(), 0
    for name, step, ckpt, scale in jobs:
        swap_adapter(network, ckpt, scale)
        for p in prompts:
            for seed in args.seeds:
                latents[(name, step, p["id"], seed)] = sample_latents(
                    transformer, embeds[p["id"]], sigmas, seed, latent_h, latent_w, dtype, device
                ).cpu()
                done += 1
        rate = (time.time() - started) / done
        print(
            f"  {name:>11s} {step:>5s}  {done}/{total}  "
            f"({rate:.1f}s/sample, ~{rate * (total - done) / 60:.0f} min left)"
        )

    del transformer, network
    gc.collect()
    torch.cuda.empty_cache()

    vae = load_qwen_image_2_1_vae(model_config["vae_path"], device=device, dtype=dtype, include_decoder=True)
    records = []
    for (name, step, prompt_id, seed), latent in latents.items():
        folder = outdir / "base" if name == "base" else outdir / name / step
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{prompt_id}_s{seed}.png"
        with torch.no_grad():
            image = vae.decode(latent.to(device, dtype))
        array = (image[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        Image.fromarray(array).save(path)
        records.append(
            {"run": name, "step": step, "prompt": prompt_id, "seed": seed, "path": path.relative_to(outdir).as_posix()}
        )

    manifest = {
        "prompts": prompts,
        "seeds": args.seeds,
        "steps": args.steps,
        "size": edge,
        "runs": {name: [step_of(c) for c in ckpts] for name, ckpts in runs},
        "dataset": configs[0]["dataset"]["path"],
        "images": records,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(records)} images and {outdir / 'manifest.json'}")


if __name__ == "__main__":
    main()
