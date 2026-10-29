"""
LTX-2 causal video VAE, encoder only (training caches the latents, previews are not rendered).

Mirrors comfy/ldm/lightricks/vae/causal_video_autoencoder.py, whole clips at a time instead of in streamed chunks
(the two are equivalent). Module names match the file (``encoder.down_blocks.N.res_blocks.M.conv1.conv.weight`` ...),
so the file's ``encoder.*`` and ``per_channel_statistics.*`` tensors load with ``strict=True``.

Pixels in [-1, 1], ``[B, 3, 1 + 8k, H, W]`` with H and W multiples of 32
-> normalised latents ``[B, 128, 1 + k, H/32, W/32]``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

STAT_NAMES = ("std-of-means", "mean-of-means")


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class CausalConv3d(nn.Module):
    """A 3D convolution that sees the past only: the first frame is repeated ``kernel - 1`` times in front."""

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv(torch.cat((x[:, :, :1].repeat(1, 1, 2, 1, 1), x), dim=2)))


class LayerNorm(nn.Module):
    """LayerNorm over the channels of a [B, C, F, H, W] tensor."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.norm(x.movedim(1, -1)).movedim(-1, 1))


def make_norm(kind: str) -> nn.Module:
    """Only the pixel norm the LTX-2.x VAEs use (a group norm would depend on how ComfyUI chunks the clip)."""
    if kind != "pixel_norm":
        raise ValueError(f"unsupported VAE norm_layer {kind!r}: the LTX-2.x video VAEs use pixel_norm")
    return PixelNorm()


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str):
        super().__init__()
        self.norm1, self.norm2 = make_norm(norm), make_norm(norm)
        self.conv1, self.conv2 = CausalConv3d(in_channels, out_channels), CausalConv3d(out_channels, out_channels)
        resize = in_channels != out_channels
        self.conv_shortcut = nn.Conv3d(in_channels, out_channels, 1) if resize else nn.Identity()
        self.norm3 = LayerNorm(in_channels, 1e-6) if resize else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return cast(torch.Tensor, h + self.conv_shortcut(self.norm3(x)))


class UNetMidBlock3D(nn.Module):
    def __init__(self, channels: int, num_layers: int, norm: str):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResnetBlock3D(channels, channels, norm) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x)
        return x


def fold(x: torch.Tensor, stride: tuple[int, int, int]) -> torch.Tensor:
    """Space-to-depth: [B, C, F p1, H p2, W p3] -> [B, C p1 p2 p3, F, H, W]."""
    b, c, f, h, w = x.shape
    p1, p2, p3 = stride
    x = x.reshape(b, c, f // p1, p1, h // p2, p2, w // p3, p3)
    return x.permute(0, 1, 3, 5, 7, 2, 4, 6).reshape(b, c * p1 * p2 * p3, f // p1, h // p2, w // p3)


class SpaceToDepthDownsample(nn.Module):
    """A strided downsample: a causal conv folded into channels, plus the group-averaged fold of the input."""

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int]):
        super().__init__()
        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = CausalConv3d(in_channels, out_channels // math.prod(stride))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride[0] == 2:
            x = torch.cat((x[:, :, :1], x), dim=2)  # the first frame stands alone: repeat it to pair the rest
        skip = fold(x, self.stride)
        skip = skip.reshape(skip.shape[0], -1, self.group_size, *skip.shape[2:]).mean(dim=2)
        return fold(cast(torch.Tensor, self.conv(x)), self.stride) + skip


COMPRESS = {
    "compress_time": (2, 1, 1), "compress_space": (1, 2, 2), "compress_all": (2, 2, 2), "compress_all_x_y": (2, 2, 2),
}  # fmt: skip
COMPRESS_RES = {"compress_time_res": (2, 1, 1), "compress_space_res": (1, 2, 2), "compress_all_res": (2, 2, 2)}


class Encoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.patch, norm = config.get("patch_size", 1), config.get("norm_layer", "pixel_norm")
        self.latent_channels = config["latent_channels"]
        channels = config.get("encoder_base_channels", 128)
        self.conv_in = CausalConv3d(config.get("in_channels", 3) * self.patch**2, channels)
        self.down_blocks = nn.ModuleList()
        for name, spec in config["encoder_blocks"]:
            params = {"num_layers": spec} if isinstance(spec, int) else spec
            grown = params.get("multiplier", 2) * channels
            if name == "res_x":
                block: nn.Module = UNetMidBlock3D(channels, params["num_layers"], norm)
            elif name == "res_x_y":
                block, channels = ResnetBlock3D(channels, grown, norm), grown
            elif name in COMPRESS_RES:
                block, channels = SpaceToDepthDownsample(channels, grown, COMPRESS_RES[name]), grown
            elif name in COMPRESS:
                out = grown if name == "compress_all_x_y" else channels
                block, channels = CausalConv3d(channels, out, COMPRESS[name]), out
            else:
                raise ValueError(f"unknown block: {name}")
            self.down_blocks.append(block)
        self.conv_norm_out = make_norm(norm)
        extra = {"per_channel": self.latent_channels, "uniform": 1, "constant": 1}.get(
            config.get("latent_log_var", "per_channel"), 0
        )
        self.conv_out = CausalConv3d(channels, self.latent_channels + extra)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, F, H, W] -> the latent means [B, latent_channels, F', H', W']."""
        b, c, f, h, w = x.shape
        p = self.patch
        x = (
            x.reshape(b, c, f, h // p, p, w // p, p)
            .permute(0, 1, 6, 4, 2, 3, 5)
            .reshape(b, c * p * p, f, h // p, w // p)
        )
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        return cast(torch.Tensor, self.conv_out(F.silu(self.conv_norm_out(x))))[:, : self.latent_channels]


class VideoVAEEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.encoder = Encoder(config)
        self.per_channel_statistics = nn.Module()
        for name in STAT_NAMES:
            self.per_channel_statistics.register_buffer(name, torch.zeros(config["latent_channels"]))

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels in [-1, 1], [B, 3, 1 + 8k (extra frames are dropped), H, W] -> normalised latents."""
        frames = max(1, 1 + ((pixels.shape[2] - 1) // 8) * 8)
        means = self.encoder(pixels[:, :, :frames])
        std, mean = (self.per_channel_statistics.get_buffer(n).view(1, -1, 1, 1, 1).to(means) for n in STAT_NAMES)
        return cast(torch.Tensor, (means - mean) / std)


def load_video_vae_encoder(
    path: str | Path, device: torch.device, dtype: torch.dtype = torch.float32
) -> VideoVAEEncoder:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LTX video VAE not found at {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if "config" not in metadata:
        raise ValueError(f"{path.name} has no config metadata; this needs ltx-2.5-video-vae-conv-bf16.safetensors")
    state = {
        k: v for k, v in load_file(str(path)).items() if k.startswith("encoder.") or k.split(".")[-1] in STAT_NAMES
    }
    model = VideoVAEEncoder(json.loads(metadata["config"])["vae"])
    info = model.load_state_dict(state, strict=False)
    if info.missing_keys:
        raise ValueError(f"{path.name} lacks encoder tensors {info.missing_keys[:5]}")
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)
