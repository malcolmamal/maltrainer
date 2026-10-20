"""
Flux2 VAE: FLUX.3's (and MOSS-Image's) latent space. The encoder caches training latents; the
decoder renders training previews.

ComfyUI's flux2_vae.safetensors is an SD-style AutoencoderKL (32 latent channels, 8x down)
with a batch-norm latent layer (comfy/ldm/models/autoencoder.py, ``batch_norm_latent``). To
encode:
1. map pixels [0, 1] -> [-1, 1]
2. encoder -> quant_conv -> the posterior mean (ComfyUI's regularizer does not sample)
3. fold each 2x2 patch into channels: 32 x (2H, 2W) -> 128 x (H, W), 16x down overall
4. normalize with the stored running mean / var (eps 1e-4, no affine)

The result is what the DiT sees: latent_formats.Flux2.process_in is the identity. Decoding runs
the same steps backwards: undo the batch norm, unfold the 2x2 patches, post_quant_conv ->
decoder, [-1, 1] -> [0, 1]. The decoder's layout follows BFL's reference autoencoder
(black-forest-labs flux2 src/flux2/autoencoder.py, Apache-2.0), with the file's own key names.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from maltrainer.models.flux3.vae_net import AttnBlock, Decoder, Downsample, Encoder, ResnetBlock, Upsample

logger = logging.getLogger(__name__)

BN_EPS = 1e-4

__all__ = [
    "BN_EPS",
    "AttnBlock",
    "Decoder",
    "Downsample",
    "Encoder",
    "Flux2VAEDecoder",
    "Flux2VAEEncoder",
    "ResnetBlock",
    "Upsample",
    "load_flux2_vae_decoder",
    "load_flux2_vae_encoder",
]


class Flux2VAEDecoder(nn.Module):
    def __init__(self, ch: int = 128, z_channels: int = 32, embed_dim: int = 32):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)
        self.decoder = Decoder(ch=ch, z_channels=z_channels)
        self.register_buffer("bn_running_mean", torch.zeros(4 * embed_dim))
        self.register_buffer("bn_running_var", torch.ones(4 * embed_dim))

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Latents [B, 128, h, w] (what the DiT sees) -> RGB [B, 3, 16h, 16w] in [0, 1]."""
        dtype = self.post_quant_conv.weight.dtype
        z = latents.to(dtype)
        mean = self.get_buffer("bn_running_mean").to(dtype).view(1, -1, 1, 1)
        std = torch.sqrt(self.get_buffer("bn_running_var").to(dtype).view(1, -1, 1, 1) + BN_EPS)
        z = z * std + mean
        B, C, h, w = z.shape
        z = z.view(B, C // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(B, C // 4, 2 * h, 2 * w)
        image = self.decoder(self.post_quant_conv(z))
        return cast(torch.Tensor, ((image.float() + 1.0) / 2.0).clamp_(0.0, 1.0))


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
        rm, rv = (self.get_buffer(n).to(z.dtype) for n in ("bn_running_mean", "bn_running_var"))
        return F.batch_norm(z, rm, rv, training=False, eps=BN_EPS)


def _read_vae_state(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Flux2 VAE not found at {path}")
    state = load_file(str(path))
    for prefix in ("first_stage_model.", "vae."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    if "bn.running_mean" not in state:
        raise ValueError(
            f"{path.name} has no batch-norm latent statistics; this needs the Flux2 VAE (flux2_vae.safetensors)"
        )
    return state


def load_flux2_vae_decoder(
    path: str | Path, device: torch.device, dtype: torch.dtype = torch.float32
) -> Flux2VAEDecoder:
    """The decoder half of ComfyUI's flux2_vae.safetensors, for previews."""
    state = _read_vae_state(path)
    if "decoder.post_quant_conv.weight" in state:
        state = {key.replace("decoder.post_quant_conv.", "post_quant_conv.", 1): value for key, value in state.items()}
    if "decoder.conv_out.weight" not in state:
        raise ValueError(f"{Path(path).name} has no decoder")
    model = Flux2VAEDecoder(
        ch=state["decoder.conv_out.weight"].shape[1],
        z_channels=state["decoder.conv_in.weight"].shape[1],
        embed_dim=state["post_quant_conv.weight"].shape[1],
    )
    decoder_state = {key: value for key, value in state.items() if key.startswith(("decoder.", "post_quant_conv."))}
    decoder_state["bn_running_mean"] = state["bn.running_mean"]
    decoder_state["bn_running_var"] = state["bn.running_var"]
    model.load_state_dict({key: value.float() for key, value in decoder_state.items()}, strict=True)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    logger.info("Flux2 VAE decoder loaded from %s (%s)", Path(path).name, dtype)
    return model


def load_flux2_vae_encoder(
    path: str | Path, device: torch.device, dtype: torch.dtype = torch.float32
) -> Flux2VAEEncoder:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Flux2 VAE not found at {path}")
    state = load_file(str(path))
    for prefix in ("first_stage_model.", "vae."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    if "bn.running_mean" not in state:
        raise ValueError(
            f"{path.name} has no batch-norm latent statistics; FLUX.3 needs the Flux2 VAE (flux2_vae.safetensors)"
        )
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
