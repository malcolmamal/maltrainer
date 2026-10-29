"""
LTX-2 causal audio VAE, encoder only.

Mirrors comfy/ldm/lightricks/vae/causal_audio_autoencoder.py and audio_vae.py: a 2D conv encoder over the log-mel
spectrogram ([B, 2, time, 64]) that is causal along time, its posterior mean, normalised with the per-channel statistics
in patch space. Module names match the file (``audio_vae.encoder.down.N.block.M.conv1.conv.weight`` ...).

Waveform ``[B, 2, samples]`` -> latents ``[B, 8, time / 4, 16]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from maltrainer.models.ltx25.audio_features import SAMPLE_RATE, log_mel, resample

PREFIX = "audio_vae."


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-6)


class CausalConv2d(nn.Module):
    """A conv that sees the past along the first spatial axis (time) only; the second axis (mel) is padded evenly."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3):
        super().__init__()
        self.pad = ((kernel - 1) // 2, (kernel - 1) - (kernel - 1) // 2, kernel - 1, 0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(F.pad(x, self.pad)))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(F.pad(x, (0, 1, 2, 0))))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1, self.norm2 = PixelNorm(), PixelNorm()
        self.conv1, self.conv2 = CausalConv2d(in_channels, out_channels), CausalConv2d(out_channels, out_channels)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))
        return cast(torch.Tensor, (self.nin_shortcut(x) if hasattr(self, "nin_shortcut") else x) + h)


class Level(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, blocks: int, downsample: bool):
        super().__init__()
        self.block = nn.ModuleList(
            [ResnetBlock(channels_in if i == 0 else channels_out, channels_out) for i in range(blocks)]
        )
        if downsample:
            self.downsample = Downsample(channels_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            x = block(x)
        return self.downsample(x) if hasattr(self, "downsample") else x


class Mid(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block_1, self.block_2 = ResnetBlock(channels, channels), ResnetBlock(channels, channels)


class Encoder(nn.Module):
    def __init__(self, ddconfig: dict[str, Any]):
        super().__init__()
        if (
            ddconfig.get("attn_resolutions")
            or ddconfig.get("mid_block_add_attention")
            or ddconfig.get("norm_type", "group") != "pixel"
        ):
            raise ValueError("only the attention-free pixel-norm audio VAE of LTX-2 is supported")
        if ddconfig.get("causality_axis", "width") != "height":
            raise ValueError("only the time-causal ('height') audio VAE of LTX-2 is supported")
        ch, mult, self.z_channels = ddconfig["ch"], ddconfig["ch_mult"], ddconfig["z_channels"]
        self.conv_in = CausalConv2d(ddconfig["in_channels"], ch)
        widths = [ch, *(ch * m for m in mult)]
        self.down = nn.ModuleList(
            [Level(widths[i], widths[i + 1], ddconfig["num_res_blocks"], i != len(mult) - 1) for i in range(len(mult))]
        )
        self.mid = Mid(widths[-1])
        self.conv_out = CausalConv2d(
            widths[-1], 2 * self.z_channels if ddconfig.get("double_z", True) else self.z_channels
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(mel)
        for level in self.down:
            x = level(x)
        x = self.mid.block_2(self.mid.block_1(x))
        return cast(torch.Tensor, self.conv_out(F.silu(PixelNorm()(x))))[:, : self.z_channels]


class AudioVAEEncoder(nn.Module):
    def __init__(self, ddconfig: dict[str, Any]):
        super().__init__()
        self.encoder = Encoder(ddconfig)
        self.in_channels = ddconfig["in_channels"]
        self.per_channel_statistics = nn.Module()
        for name in ("std-of-means", "mean-of-means"):
            self.per_channel_statistics.register_buffer(
                name, torch.zeros(self.encoder.z_channels * ddconfig["mel_bins"] // 4)
            )

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Waveform [B, 1 or 2, samples] -> normalised latents [B, z, frames, mel / 4]."""
        if waveform.shape[1] != self.in_channels:
            if waveform.shape[1] != 1:
                raise ValueError(f"audio must have {self.in_channels} channels, got {waveform.shape[1]}")
            waveform = waveform.expand(-1, self.in_channels, -1)
        latents = self.encoder(log_mel(resample(waveform, sample_rate, SAMPLE_RATE)))
        batch, channels, frames, bins = latents.shape
        patched = latents.permute(0, 2, 1, 3).reshape(batch, frames, channels * bins)
        stats = self.per_channel_statistics
        patched = (patched - stats.get_buffer("mean-of-means").to(patched)) / stats.get_buffer("std-of-means").to(
            patched
        )
        return cast(torch.Tensor, patched.reshape(batch, frames, channels, bins).permute(0, 2, 1, 3))


def load_audio_vae_encoder(
    path: str | Path, device: torch.device, dtype: torch.dtype = torch.float32
) -> AudioVAEEncoder:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LTX audio VAE not found at {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if "config" not in metadata:
        raise ValueError(f"{path.name} has no config metadata; this needs ltx-2.5-audio-vae-bf16.safetensors")
    params = json.loads(metadata["config"])["audio_vae"]["model"]["params"]
    ddconfig = params.get("ddconfig", params.get("encoder"))
    state = {k[len(PREFIX) :]: v for k, v in load_file(str(path)).items() if k.startswith(PREFIX)}
    wanted = {
        k: v for k, v in state.items() if k.startswith("encoder.") or k.endswith(("std-of-means", "mean-of-means"))
    }
    model = AudioVAEEncoder(ddconfig)
    info = model.load_state_dict(wanted, strict=False)
    if info.missing_keys:
        raise ValueError(f"{path.name} lacks audio encoder tensors {info.missing_keys[:5]}")
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)
