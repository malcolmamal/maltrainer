"""
MiniMax H3 audio VAE -- encoder side only.

A DAC-lineage waveform encoder followed by an attention projection down to 32 latent
channels. Training only ever needs ``encode``, so the BigVGAN decoder is not built and its
tensors are skipped on load.

Module layout mirrors comfy/ldm/minimax/audio_vae.py exactly, so ComfyUI's single-file
``minimax_h3_audio_vae_fp32.safetensors`` loads by name. The original release
(``FL2VA/audio_vae/model.safetensors``) stores weight-norm ``weight_g``/``weight_v`` pairs
instead of plain weights; those are folded on load.

Shapes (ComfyUI convention):
    waveform [B, 2, L] at 32 kHz, in [-1, 1]  ->  normalized latents [B, 32, 2, T]
    T = ceil(L / 800), i.e. 40 latent frames per second. Each stereo channel goes through
    the mono encoder as its own batch item.

The layers live in ``audio_vae_layers`` and ``audio_vae_attention``, the constants in ``audio_vae_constants``; every
name is re-exported here.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from maltrainer.models.minimax_h3.audio_vae_attention import AttnProjection, CausalAttention, GeGluMlp
from maltrainer.models.minimax_h3.audio_vae_constants import (
    DEFAULT_LATENTS_MEAN,
    DEFAULT_LATENTS_STD,
    HOP_LENGTH,
    LATENT_CHANNELS,
    LATENTS_PER_SECOND,
    SAMPLE_RATE,
)
from maltrainer.models.minimax_h3.audio_vae_layers import DacEncoder, EncoderBlock, ResidualUnit, Snake1d, _snake

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LATENTS_MEAN",
    "DEFAULT_LATENTS_STD",
    "HOP_LENGTH",
    "LATENTS_PER_SECOND",
    "LATENT_CHANNELS",
    "SAMPLE_RATE",
    "AttnProjection",
    "CausalAttention",
    "DacEncoder",
    "EncoderBlock",
    "GeGluMlp",
    "MiniMaxH3AudioEncoder",
    "ResidualUnit",
    "Snake1d",
    "_snake",
    "load_minimax_h3_audio_encoder",
]


class MiniMaxH3AudioEncoder(nn.Module):
    """Stereo 32 kHz waveform -> normalized H3 audio latents. Frozen, fp32."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = DacEncoder(64, (2, 4, 4, 5, 5), 2048)
        self.pre_block = AttnProjection(2048, LATENT_CHANNELS, num_heads=8)
        self.mean_proj = nn.Conv1d(LATENT_CHANNELS, LATENT_CHANNELS, 1)
        self.register_buffer("latents_mean", torch.tensor(DEFAULT_LATENTS_MEAN, dtype=torch.float32))
        self.register_buffer("latents_std", torch.tensor(DEFAULT_LATENTS_STD, dtype=torch.float32))

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """[B, 2, L] at 32 kHz -> normalized latents [B, 32, 2, ceil(L / 800)].

        The tail is zero-padded to a whole latent frame and the posterior mean is used
        (no sampling), as in ComfyUI and the released pipeline.
        """
        b, s, length = waveform.shape
        waveform = waveform.float()
        right_pad = math.ceil(length / HOP_LENGTH) * HOP_LENGTH - length
        x = F.pad(waveform, (0, right_pad)).reshape(b * s, 1, -1)
        x = self.encoder(x)
        x = self.pre_block(x.transpose(1, 2)).transpose(1, 2)
        z = self.mean_proj(x)
        z = (z - self.get_buffer("latents_mean").view(1, -1, 1)) / self.get_buffer("latents_std").view(1, -1, 1)
        return cast(torch.Tensor, z.reshape(b, s, z.shape[1], z.shape[2]).permute(0, 2, 1, 3))


def _fold_weight_norm(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold weight_g/weight_v pairs (nn.utils.weight_norm, dim=0) into plain weights."""
    folded = {}
    for name, tensor in state_dict.items():
        if name.endswith(".weight_g"):
            continue
        if name.endswith(".weight_v"):
            stem = name[: -len("weight_v")]
            g = state_dict[stem + "weight_g"]
            norm = tensor.norm(p=2, dim=tuple(range(1, tensor.ndim)), keepdim=True)
            folded[stem + "weight"] = g * tensor / norm
        else:
            folded[name] = tensor
    return folded


_ENCODER_PREFIXES = ("encoder.", "pre_block.", "mean_proj.", "latents_mean", "latents_std")


def load_minimax_h3_audio_encoder(path: str | Path, device: torch.device) -> MiniMaxH3AudioEncoder:
    """Load the audio VAE encoder from ComfyUI's single file or the original release."""
    from safetensors.torch import load_file

    state = load_file(str(path))
    if any(key.endswith(".weight_g") for key in state):
        state = _fold_weight_norm(state)
    state = {key: value.float() for key, value in state.items() if key.startswith(_ENCODER_PREFIXES)}
    model = MiniMaxH3AudioEncoder()
    has_stats = "latents_mean" in state and "latents_std" in state
    info = model.load_state_dict(state, strict=False)
    missing = [key for key in info.missing_keys if key not in ("latents_mean", "latents_std")]
    if missing:
        raise ValueError(
            f"{Path(path).name} did not supply {len(missing)} audio VAE encoder tensor(s); "
            f"examples: {missing[:5]}. Is it the MiniMax H3 audio VAE?"
        )
    if info.unexpected_keys:
        raise ValueError(
            f"{Path(path).name} has {len(info.unexpected_keys)} encoder tensor(s) this module does "
            f"not expect; examples: {info.unexpected_keys[:5]}"
        )
    if not has_stats:
        logger.info("%s carries no latent statistics; using the released config values.", Path(path).name)

    # fp32 only: the reference pins this VAE to fp32, and the stats span 1.5-3.3.
    model.to(device=device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return model
