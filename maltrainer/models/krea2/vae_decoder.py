"""
Wan 2.1 VAE decoder for still images: Krea 2's latents back to pixels, for training previews.

The decoder half of the same causal 3D network as ``vae.py``'s encoder (ComfyUI comfy/ldm/wan/vae.py Decoder3d),
reduced to one frame the same way: every causal 3x3x3 convolution sees two zero frames in front of it and becomes a
2D convolution with its kernel's last temporal tap. A single latent frame decodes without a feature cache, and
without one the upsamplers' temporal ``time_conv`` never runs (ComfyUI skips it there too), so the reduction is exact.
Loads ComfyUI's ``wan_2.1_vae.safetensors`` / ``qwen_image_vae.safetensors`` by its original key names.

Latents come in normalized (``vae.py``'s output) and go out as pixels in [0, 1]: un-normalized with the Wan21 stats
(``z * std + mean``), through the root ``conv2`` (post-quant conv) and the decoder, then ``(x + 1) / 2`` clamped,
as ComfyUI's VAE decode.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from torch import nn

from maltrainer.models.krea2.vae import AttentionBlock, CausalConv2d, ResidualBlock, RMSNorm
from maltrainer.models.krea2.vae_constants import LATENTS_MEAN, LATENTS_STD

logger = logging.getLogger(__name__)


class Upsample(nn.Module):
    """upsample2d / upsample3d on one frame: nearest-exact 2x, then a 3x3 conv that halves the channels."""

    def __init__(self, dim: int):
        super().__init__()
        self.resample = nn.Sequential(
            nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.resample(x))


class WanImageDecoder(nn.Module):
    def __init__(
        self,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        image_channels: int = 3,
    ) -> None:
        super().__init__()
        dims = [dim * u for u in (dim_mult[-1], *reversed(dim_mult))]
        self.conv1 = CausalConv2d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0]), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0])
        )
        layers: list[nn.Module] = []
        for i, (stage_in, out_dim) in enumerate(itertools.pairwise(dims)):
            in_dim = stage_in // 2 if i > 0 else stage_in  # the upsampler before this stage halved the channels
            for j in range(num_res_blocks + 1):
                layers.append(ResidualBlock(in_dim if j == 0 else out_dim, out_dim))
            if i != len(dim_mult) - 1:
                layers.append(Upsample(out_dim))
        self.upsamples = nn.Sequential(*layers)
        self.head = nn.Sequential(
            RMSNorm(out_dim, images=False), nn.SiLU(), CausalConv2d(out_dim, image_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(self.upsamples(self.middle(self.conv1(x)))))


class Krea2VAEDecoder(nn.Module):
    """The root ``conv2`` (post-quant conv) and the decoder half of the Wan 2.1 VAE, with the Wan21 stats."""

    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(self, dim: int = 96, z_dim: int = 16):
        super().__init__()
        self.conv2 = CausalConv2d(z_dim, z_dim, 1)
        self.decoder = WanImageDecoder(dim=dim, z_dim=z_dim)
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN).view(1, -1, 1, 1), persistent=False)
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD).view(1, -1, 1, 1), persistent=False)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalized latents [B, 16, h, w] -> pixels [B, 3, 8h, 8w] in [0, 1], fp32."""
        z = latents.float() * self.latents_std.to(latents.device) + self.latents_mean.to(latents.device)
        out = cast(torch.Tensor, self.decoder(self.conv2(z.to(self.conv2.weight.dtype)))).float()
        return ((out + 1.0) / 2.0).clamp(0.0, 1.0)


def load_krea2_vae_decoder(
    path: str | Path, device: torch.device, dtype: torch.dtype = torch.float32
) -> Krea2VAEDecoder:
    """The decoder from ComfyUI's Wan 2.1 VAE file (the encoder tensors are not read)."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        state = {
            key: handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith(("decoder.", "conv2.")) and ".time_conv." not in key
        }
    dim = int(state["decoder.head.2.weight"].shape[1]) if "decoder.head.2.weight" in state else 96
    model = Krea2VAEDecoder(dim=dim)
    info = model.load_state_dict(state, strict=False)
    if info.missing_keys or info.unexpected_keys:
        raise ValueError(
            f"{Path(path).name} is not a Wan 2.1 VAE this decoder can load "
            f"(missing {info.missing_keys[:5]}, unexpected {info.unexpected_keys[:5]})"
        )
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    logger.info("Krea 2 VAE decoder from %s (%s)", Path(path).name, str(dtype).replace("torch.", ""))
    return model
