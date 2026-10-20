"""SD-VAE-FT-MSE encode path for Supra2-IMG. The model sees latents scaled by 0.18215."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from diffusers import AutoencoderKL
from safetensors.torch import load_file

from maltrainer.models.supra2_img.model import VAE_SCALE


def load_sd_vae(
    weights_path: str | Path, config_path: str | Path, device: torch.device, dtype: torch.dtype
) -> AutoencoderKL:
    with Path(config_path).open(encoding="utf-8") as handle:
        vae = AutoencoderKL.from_config(json.load(handle))
    state = load_file(str(weights_path), device="cpu")
    vae._fix_state_dict_keys_on_load(state)
    vae.load_state_dict(state, strict=True)
    vae.to(device=device, dtype=dtype)
    vae.requires_grad_(False)
    vae.eval()
    return cast(AutoencoderKL, vae)


@torch.no_grad()
def encode_latents(vae: AutoencoderKL, pixels_01: torch.Tensor) -> torch.Tensor:
    """Encode RGB pixels in [0, 1] to the scaled latent the DiT was trained on."""
    pixels = pixels_01 * 2.0 - 1.0
    posterior = cast(Any, vae).encode(pixels).latent_dist
    return cast(torch.Tensor, posterior.mode() * VAE_SCALE)
