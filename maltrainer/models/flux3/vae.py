"""
Flux2 VAE encoder: FLUX.3's latent space.

ComfyUI's flux2_vae.safetensors is an SD-style AutoencoderKL (32 latent channels, 8x down)
with a batch-norm latent layer (comfy/ldm/models/autoencoder.py, ``batch_norm_latent``). To
encode:
1. map pixels [0, 1] -> [-1, 1]
2. encoder -> quant_conv -> the posterior mean (ComfyUI's regularizer does not sample)
3. fold each 2x2 patch into channels: 32 x (2H, 2W) -> 128 x (H, W), 16x down overall
4. normalize with the stored running mean / var (eps 1e-4, no affine)

The result is what the DiT sees: latent_formats.Flux2.process_in is the identity. Only the
encoder is built, since training never decodes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Union

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

logger = logging.getLogger(__name__)

BN_EPS = 1e-4


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = _norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if hasattr(self, "nin_shortcut"):
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _norm(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        B, C, H, W = h.shape
        q, k, v = (t.view(B, 1, C, H * W).transpose(2, 3) for t in (self.q(h), self.k(h), self.v(h)))
        out = F.scaled_dot_product_attention(q, k, v).transpose(2, 3).reshape(B, C, H, W)
        return x + self.proj_out(out)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Encoder(nn.Module):
    def __init__(self, ch: int = 128, ch_mult: Sequence[int] = (1, 2, 4, 4), num_res_blocks: int = 2, z_channels: int = 32):
        super().__init__()
        self.conv_in = nn.Conv2d(3, ch, 3, padding=1)
        in_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for level, mult in enumerate(ch_mult):
            down = nn.Module()
            block_in, block_out = ch * in_mult[level], ch * mult
            down.block = nn.ModuleList()
            for _ in range(num_res_blocks):
                down.block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            if level != len(ch_mult) - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)
        self.norm_out = _norm(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for down in self.down:
            for block in down.block:
                h = block(h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.conv_out(F.silu(self.norm_out(h)))


class Flux2VAEEncoder(nn.Module):
    def __init__(self, ch: int = 128, z_channels: int = 32, embed_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(ch=ch, z_channels=z_channels)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * embed_dim, 1)
        self.register_buffer("bn_running_mean", torch.zeros(4 * embed_dim))
        self.register_buffer("bn_running_var", torch.ones(4 * embed_dim))

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """RGB [B, 3, H, W] in [0, 1], sides multiples of 16 -> latents [B, 128, H/16, W/16]."""
        dtype = self.quant_conv.weight.dtype
        moments = self.quant_conv(self.encoder(images.to(dtype) * 2.0 - 1.0))
        mean = moments.chunk(2, dim=1)[0]
        B, C, H, W = mean.shape
        z = mean.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(B, C * 4, H // 2, W // 2)
        return F.batch_norm(z, self.bn_running_mean.to(z.dtype), self.bn_running_var.to(z.dtype), training=False, eps=BN_EPS)


def load_flux2_vae_encoder(path: Union[str, Path], device: torch.device, dtype: torch.dtype = torch.float32) -> Flux2VAEEncoder:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Flux2 VAE not found at {path}")
    state = load_file(str(path))
    for prefix in ("first_stage_model.", "vae."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    if "bn.running_mean" not in state:
        raise ValueError(f"{path.name} has no batch-norm latent statistics; FLUX.3 needs the Flux2 VAE (flux2_vae.safetensors)")
    if "encoder.quant_conv.weight" in state:
        state = {key.replace("encoder.quant_conv.", "quant_conv.", 1): value for key, value in state.items()}

    z_channels = state["decoder.conv_in.weight"].shape[1] if "decoder.conv_in.weight" in state else 32
    model = Flux2VAEEncoder(
        ch=state["encoder.conv_in.weight"].shape[0],
        z_channels=z_channels,
        embed_dim=state["quant_conv.weight"].shape[0] // 2,
    )
    encoder_state = {key: value for key, value in state.items() if key.startswith(("encoder.", "quant_conv."))}
    encoder_state["bn_running_mean"] = state["bn.running_mean"]
    encoder_state["bn_running_var"] = state["bn.running_var"]
    model.load_state_dict({key: value.float() for key, value in encoder_state.items()}, strict=True)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    logger.info("Flux2 VAE encoder loaded from %s (%s)", path.name, dtype)
    return model
