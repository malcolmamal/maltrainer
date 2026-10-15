"""
Wan 2.1 VAE encoder for still images -- Krea 2's latent space.

Krea 2 uses the Wan 2.1 VAE (ComfyUI latent format ``Wan21``: 16 channels, 8x spatial,
per-channel mean/std). The VAE is a causal 3D network, but a still is a single frame: every
causal 3x3x3 convolution then sees two zero frames in front of it, so it reduces exactly to a
2D convolution with the kernel's last temporal tap, and the temporal ``time_conv`` of the
downsamplers never runs (ComfyUI skips it for the first chunk as well). This module is that
2D reduction, loading ComfyUI's ``wan_2.1_vae.safetensors`` by its original key names.

Pixels go in as [0, 1]; ComfyUI maps them to [-1, 1] before encoding and normalizes the
posterior mean with ``(z - mean) / std`` (Wan21.process_in, scale_factor 1.0).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

logger = logging.getLogger(__name__)

# comfy/latent_formats.py Wan21
LATENTS_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)
LATENTS_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)


class CausalConv2d(nn.Module):
    """A CausalConv3d applied to one frame: the last temporal tap of its kernel as a 2D conv."""

    def __init__(self, in_dim: int, out_dim: int, kernel: int, padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, kernel, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight[:, :, -1].to(x.dtype), self.bias.to(x.dtype), padding=self.padding)


class RMSNorm(nn.Module):
    """Wan RMS_norm: F.normalize over channels * sqrt(C) * gamma."""

    def __init__(self, dim: int, images: bool):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1) if images else (dim, 1, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.reshape(-1, 1, 1).to(x.dtype)
        return F.normalize(x, dim=1) * self.scale * gamma


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm(in_dim, images=False), nn.SiLU(), CausalConv2d(in_dim, out_dim, 3, padding=1),
            RMSNorm(out_dim, images=False), nn.SiLU(), nn.Dropout(0.0), CausalConv2d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = (t.reshape(b, 1, c, h * w).transpose(-1, -2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(b, c, h, w)
        return self.proj(out) + identity


class Resample(nn.Module):
    """downsample2d / downsample3d. For a single frame both are the same 2D strided conv."""

    def __init__(self, dim: int, temporal: bool):
        super().__init__()
        self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=2))
        self.temporal = temporal

    def forward(self, x):
        return self.resample(x)


class WanImageEncoder(nn.Module):
    def __init__(self, dim: int = 96, z_dim: int = 16, dim_mult=(1, 2, 4, 4), num_res_blocks: int = 2,
                 temporal_downsample=(False, True, True), image_channels: int = 3):
        super().__init__()
        dims = [dim * u for u in (1,) + tuple(dim_mult)]
        self.conv1 = CausalConv2d(image_channels, dims[0], 3, padding=1)
        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                layers.append(Resample(out_dim, temporal_downsample[i]))
        self.downsamples = nn.Sequential(*layers)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim))
        self.head = nn.Sequential(RMSNorm(out_dim, images=False), nn.SiLU(), CausalConv2d(out_dim, z_dim * 2, 3, padding=1))

    def forward(self, x):
        return self.head(self.middle(self.downsamples(self.conv1(x))))


class Krea2VAE(nn.Module):
    """Encoder half of the Wan 2.1 VAE plus its root ``conv1`` (quant conv) and the Wan21 stats."""

    def __init__(self, dim: int = 96, z_dim: int = 16):
        super().__init__()
        self.encoder = WanImageEncoder(dim=dim, z_dim=z_dim)
        self.conv1 = CausalConv2d(z_dim * 2, z_dim * 2, 1)
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD).view(1, -1, 1, 1), persistent=False)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] in [0, 1] -> normalized latents [B, 16, H/8, W/8] (posterior mean)."""
        dtype = self.conv1.weight.dtype
        x = (images * 2.0 - 1.0).to(dtype)
        mu = self.conv1(self.encoder(x)).chunk(2, dim=1)[0].float()
        return (mu - self.latents_mean.to(mu.device)) / self.latents_std.to(mu.device)


def load_krea2_vae(path: Union[str, Path], device: torch.device, dtype: torch.dtype = torch.float32) -> Krea2VAE:
    """Load the encoder from ComfyUI's wan_2.1_vae.safetensors (decoder tensors are ignored)."""
    state = load_file(str(path))
    dim = int(state["encoder.conv1.weight"].shape[0]) if "encoder.conv1.weight" in state else 96
    model = Krea2VAE(dim=dim)
    wanted = {
        key: value for key, value in state.items()
        if (key.startswith("encoder.") or key.startswith("conv1.")) and ".time_conv." not in key
    }
    info = model.load_state_dict(wanted, strict=False)
    if info.missing_keys or info.unexpected_keys:
        raise ValueError(
            f"{Path(path).name} is not a Wan 2.1 VAE this encoder can load "
            f"(missing {info.missing_keys[:5]}, unexpected {info.unexpected_keys[:5]})"
        )
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return model
