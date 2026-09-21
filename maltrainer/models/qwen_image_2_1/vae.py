"""
Qwen-Image-2.1 3D VAE Encoder and Decoder.
Supports 16x spatial downsampling, 64 latent channels, and per-channel mean/std normalization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import safetensors.torch
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Official 64-channel normalization constants for Qwen-Image-2.1
LATENTS_MEAN = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
    1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
    -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376,
    -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
    0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
]

LATENTS_STD = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910,
    3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756,
    3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467,
    3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125,
    3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875,
    3.9624, 4.0168, 3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959,
    3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
]


class RMSNorm3d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = F.normalize(x.float(), dim=1, eps=self.eps).to(x.dtype)
        gamma = self.gamma.view(1, -1, 1, 1)
        return norm * (self.gamma.shape[0] ** 0.5) * gamma


class RMSNormAttn(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = F.normalize(x.float(), dim=1, eps=self.eps).to(x.dtype)
        gamma = self.gamma.view(1, -1, 1, 1)
        return norm * (self.gamma.shape[0] ** 0.5) * gamma


class CausalConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.squeeze(2)
        return F.conv2d(x, w, self.bias, stride=self.stride, padding=self.padding)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm3d(in_channels),
            nn.SiLU(),
            CausalConv3d(in_channels, out_channels, 3, padding=1),
            RMSNorm3d(out_channels),
            nn.SiLU(),
            nn.Dropout(0.0),
            CausalConv3d(out_channels, out_channels, 3, padding=1),
        )
        if in_channels != out_channels:
            self.shortcut = CausalConv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.residual(x)


class Resample2D(nn.Module):
    def __init__(self, channels: int, mode: str = "down"):
        super().__init__()
        self.mode = mode
        if mode == "down":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(channels, channels, 3, stride=(2, 2), padding=0),
            )
        else:
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest"),
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x)


class Resample3D(nn.Module):
    def __init__(self, channels: int, mode: str = "down"):
        super().__init__()
        self.mode = mode
        if mode == "down":
            self.time_conv = nn.Conv3d(channels, channels, kernel_size=1)
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(channels, channels, 3, stride=(2, 2), padding=0),
            )
        else:
            self.time_conv = nn.Conv3d(channels, channels * 2, kernel_size=1)
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest"),
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resample(x)


class VAEAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = RMSNormAttn(channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.to_qkv(norm_x).view(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).view(b, c, h, w)
        return x + self.proj(attn)


class AvgDown3D(nn.Module):
    """Parameter-free average-pool shortcut around a whole encoder stage.

    Carries no weights, so its absence is invisible to a strict state-dict load -- but the
    stage output is ``blocks(x) + avg_shortcut(x)``, and dropping it changes the latents.
    Images are a single frame, so a temporal factor averages the frame against one zero-pad
    frame; kept explicit here to stay comparable with the reference implementation.
    """

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        factor = factor_t * factor_s * factor_s
        if in_channels * factor % out_channels:
            raise ValueError(
                f"in_channels ({in_channels}) x factor ({factor}) must divide out_channels ({out_channels})"
            )
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor
        self.group_size = in_channels * factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)  # [B, C, 1, H, W]
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.view(b, c, t // ft, ft, h // fs, fs, w // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, c * self.factor, t // ft, h // fs, w // fs)
        x = x.view(b, self.out_channels, self.group_size, t // ft, h // fs, w // fs)
        return x.mean(dim=2).squeeze(2)


class DupUp3D(nn.Module):
    """Parameter-free duplicating shortcut around a whole decoder stage (mirror of AvgDown3D)."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        factor = factor_t * factor_s * factor_s
        if out_channels * factor % in_channels:
            raise ValueError(
                f"out_channels ({out_channels}) x factor ({factor}) must divide in_channels ({in_channels})"
            )
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.repeats = out_channels * factor // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)  # [B, C, 1, H, W]
        x = x.repeat_interleave(self.repeats, dim=1)
        ft, fs = self.factor_t, self.factor_s
        x = x.view(x.size(0), self.out_channels, ft, fs, fs, x.size(2), x.size(3), x.size(4))
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * ft, x.size(4) * fs, x.size(6) * fs)
        # first_chunk: a single frame keeps only the last of the duplicated frames
        return x[:, :, ft - 1:].squeeze(2)


class EncoderStage(nn.Module):
    def __init__(
        self,
        res_blocks: List[nn.Module],
        downsample: Optional[nn.Module] = None,
        shortcut: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.downsamples = nn.ModuleList(res_blocks)
        if downsample is not None:
            self.downsamples.append(downsample)
        self.avg_shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for m in self.downsamples:
            x = m(x)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(residual)
        return x


class DecoderStage(nn.Module):
    def __init__(
        self,
        res_blocks: List[nn.Module],
        upsample: Optional[nn.Module] = None,
        shortcut: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.upsamples = nn.ModuleList(res_blocks)
        if upsample is not None:
            self.upsamples.append(upsample)
        self.avg_shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for m in self.upsamples:
            x = m(x)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(residual)
        return x


class Qwen21VAEEncoder(nn.Module):
    def __init__(self, base_dim: int = 96, z_dim: int = 64):
        super().__init__()
        self.conv1 = CausalConv3d(4, base_dim, 3, padding=1)
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8, base_dim * 8]
        # temperal_downsample = [False, True, True, True]; the last stage does not downsample
        self.downsamples = nn.ModuleList([
            EncoderStage([ResBlock(dims[0], dims[0]), ResBlock(dims[0], dims[0])], Resample2D(dims[0], "down"),
                         AvgDown3D(dims[0], dims[0], factor_t=1, factor_s=2)),
            EncoderStage([ResBlock(dims[0], dims[1]), ResBlock(dims[1], dims[1])], Resample3D(dims[1], "down"),
                         AvgDown3D(dims[0], dims[1], factor_t=2, factor_s=2)),
            EncoderStage([ResBlock(dims[1], dims[2]), ResBlock(dims[2], dims[2])], Resample3D(dims[2], "down"),
                         AvgDown3D(dims[1], dims[2], factor_t=2, factor_s=2)),
            EncoderStage([ResBlock(dims[2], dims[3]), ResBlock(dims[3], dims[3])], Resample3D(dims[3], "down"),
                         AvgDown3D(dims[2], dims[3], factor_t=2, factor_s=2)),
            EncoderStage([ResBlock(dims[3], dims[4]), ResBlock(dims[4], dims[4])], None,
                         AvgDown3D(dims[3], dims[4], factor_t=1, factor_s=1)),
        ])
        self.middle = nn.ModuleList([
            ResBlock(dims[4], dims[4]),
            VAEAttention(dims[4]),
            ResBlock(dims[4], dims[4]),
        ])
        self.head = nn.Sequential(
            RMSNorm3d(dims[4]),
            nn.SiLU(),
            CausalConv3d(dims[4], z_dim * 2, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        for stage in self.downsamples:
            x = stage(x)
        for m in self.middle:
            x = m(x)
        x = self.head(x)
        return x


class Qwen21VAEDecoder(nn.Module):
    def __init__(self, z_dim: int = 64):
        super().__init__()
        self.conv1 = CausalConv3d(z_dim, 1152, 3, padding=1)
        self.middle = nn.ModuleList([
            ResBlock(1152, 1152),
            VAEAttention(1152),
            ResBlock(1152, 1152),
        ])
        # temperal_upsample = [True, True, True, False]; the last stage does not upsample
        self.upsamples = nn.ModuleList([
            DecoderStage([ResBlock(1152, 1152), ResBlock(1152, 1152), ResBlock(1152, 1152)], Resample3D(1152, "up"),
                         DupUp3D(1152, 1152, factor_t=2, factor_s=2)),
            DecoderStage([ResBlock(1152, 1152), ResBlock(1152, 1152), ResBlock(1152, 1152)], Resample3D(1152, "up"),
                         DupUp3D(1152, 1152, factor_t=2, factor_s=2)),
            DecoderStage([ResBlock(1152, 576), ResBlock(576, 576), ResBlock(576, 576)], Resample3D(576, "up"),
                         DupUp3D(1152, 576, factor_t=2, factor_s=2)),
            DecoderStage([ResBlock(576, 288), ResBlock(288, 288), ResBlock(288, 288)], Resample2D(288, "up"),
                         DupUp3D(576, 288, factor_t=1, factor_s=2)),
            DecoderStage([ResBlock(288, 144), ResBlock(144, 144), ResBlock(144, 144)], None, None),
        ])
        self.head = nn.Sequential(
            RMSNorm3d(144),
            nn.SiLU(),
            CausalConv3d(144, 4, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.conv1(z)
        for m in self.middle:
            x = m(x)
        for stage in self.upsamples:
            x = stage(x)
        x = self.head(x)
        return x


class QwenImage21VAE(nn.Module):
    """Qwen-Image-2.1 Complete VAE Wrapper."""

    def __init__(self, include_decoder: bool = False, z_dim: int = 64):
        super().__init__()
        self.encoder = Qwen21VAEEncoder(z_dim=z_dim)
        self.decoder = Qwen21VAEDecoder(z_dim=z_dim) if include_decoder else None

        # The checkpoint keeps these at its root as `conv1` / `conv2`: the encoder head's
        # moments go through quant_conv before the mean/logvar split, and latents go through
        # post_quant_conv before the decoder. Skipping them puts the latents in a different
        # space than the one the transformer was trained on.
        self.quant_conv = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = CausalConv3d(z_dim, z_dim, 1) if include_decoder else None

        self.register_buffer(
            "latents_mean",
            torch.tensor(LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latents_std",
            torch.tensor(LATENTS_STD, dtype=torch.float32).view(1, -1, 1, 1),
            persistent=False,
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize latents using channel-wise mean and std."""
        mean = self.latents_mean.to(device=latents.device, dtype=latents.dtype)
        std = self.latents_std.to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) / std

    def unnormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Unnormalize latents using channel-wise mean and std."""
        mean = self.latents_mean.to(device=latents.device, dtype=latents.dtype)
        std = self.latents_std.to(device=latents.device, dtype=latents.dtype)
        return latents * std + mean

    def encode(self, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Encode images [B, 3, H, W] into normalized latents [B, 64, H/16, W/16].

        Input images should be in range [-1.0, 1.0] or [0.0, 1.0].
        Note: If in [0, 1], input is scaled to [-1, 1] internally.
        """
        if x.min() >= 0.0 and x.max() <= 1.0:
            x = x * 2.0 - 1.0

        # VAE conv1 expects 4 input channels [B, 4, H, W] (RGBA in [-1, 1], opaque alpha = 1.0)
        if x.shape[1] == 3:
            pad = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        moments = self.quant_conv(self.encoder(x))
        mean, logvar = moments.chunk(2, dim=1)

        if deterministic:
            latents = mean
        else:
            std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))
            latents = mean + std * torch.randn_like(mean)

        return self.normalize_latents(latents)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode normalized latents [B, 64, H/16, W/16] into images [B, 3, H, W] in range [0, 1]."""
        if self.decoder is None:
            raise RuntimeError("Decoder was not loaded for this VAE instance.")
        z = self.unnormalize_latents(z)
        out = self.decoder(self.post_quant_conv(z))
        # Take first 3 channels (RGB)
        rgb = out[:, :3]
        # Rescale from [-1, 1] to [0, 1]
        rgb = ((rgb + 1.0) / 2.0).clamp(0.0, 1.0)
        return rgb


def load_qwen_image_2_1_vae(
    vae_path: Union[str, Path],
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    include_decoder: bool = False,
) -> QwenImage21VAE:
    """Load Qwen-Image-2.1 VAE from safetensors checkpoint."""
    vae_path = Path(vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")

    logger.info("Loading Qwen-Image-2.1 VAE from %s...", vae_path)
    model = QwenImage21VAE(include_decoder=include_decoder)

    with safetensors.safe_open(str(vae_path), framework="pt") as f:
        # Load encoder weights
        enc_sd = {}
        for k in f.keys():
            if k.startswith("encoder."):
                enc_sd[k[len("encoder."):]] = f.get_tensor(k)
        model.encoder.load_state_dict(enc_sd, strict=True)

        # Root-level `conv1` / `conv2` are quant_conv / post_quant_conv, not encoder layers.
        quant_sd = {k[len("conv1."):]: f.get_tensor(k) for k in f.keys() if k.startswith("conv1.")}
        if not quant_sd:
            raise KeyError(f"VAE checkpoint {vae_path} has no root `conv1.*` (quant_conv) weights")
        model.quant_conv.load_state_dict(quant_sd, strict=True)

        if include_decoder:
            dec_sd = {}
            for k in f.keys():
                if k.startswith("decoder."):
                    dec_sd[k[len("decoder."):]] = f.get_tensor(k)
            model.decoder.load_state_dict(dec_sd, strict=True)

            post_sd = {k[len("conv2."):]: f.get_tensor(k) for k in f.keys() if k.startswith("conv2.")}
            if not post_sd:
                raise KeyError(f"VAE checkpoint {vae_path} has no root `conv2.*` (post_quant_conv) weights")
            model.post_quant_conv.load_state_dict(post_sd, strict=True)

    model.requires_grad_(False)
    model.to(device=device, dtype=dtype)
    model.eval()
    logger.info("Loaded Qwen-Image-2.1 VAE successfully on %s in %s", device, dtype)
    return model
