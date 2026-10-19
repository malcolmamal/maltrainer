"""
MiniMax H3 Video VAE wrapper and encoding logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

# Per-channel latent statistics of the MiniMax H3 video VAE, from Hugging Face
# Diffusers PR #14355 (commit abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc). These are the
# same constants ComfyUI and musubi-tuner normalize with, so training latents must use
# them verbatim -- guessing them per-channel mis-scales x0 by up to 3.6x and the LoRA
# then bakes in a bogus per-channel correction that shows up as burn/artifacts.
DEFAULT_LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)

DEFAULT_LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608886,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.4498890042304992,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235595,
    3.0496184825897215,
    2.1088054180145265,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int, int],
        stride: int | Tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(
                hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode
            )
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class MiniMaxH3VideoGroupNorm(nn.GroupNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(num_layers)
            ]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList(
                [
                    MiniMaxH3VideoDownsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3VideoMidBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for _ in range(2)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoderModel(nn.Module):
    """MiniMax H3 Video VAE Encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: Tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: Tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
        latents_mean: Tuple[float, ...] = DEFAULT_LATENTS_MEAN,
        latents_std: Tuple[float, ...] = DEFAULT_LATENTS_STD,
    ) -> None:
        super().__init__()
        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    in_channels=block_in_channels[i],
                    out_channels=block_out_channels[i],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(len(block_out_channels))
            ]
        )

        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1],
            2 * latent_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, 1)

        self.register_buffer("latents_mean", torch.tensor(latents_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("latents_std", torch.tensor(latents_std, dtype=torch.float32), persistent=False)

    def encode_moments(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels shape: [B, 3, T, H, W] -> moments: [B, 48, T_latent, H//16, W//16]"""
        pixel_mean = pixels.new_tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1, 1)
        pixel_std = pixels.new_tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1, 1)
        hidden_states = (pixels.float() - pixel_mean) / pixel_std
        hidden_states = hidden_states.to(dtype=self.conv_in.weight.dtype)

        hidden_states = self.conv_in(hidden_states)
        for block in self.down_blocks:
            hidden_states = block(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        hidden_states = self.conv_out(hidden_states)
        return self.quant_conv(hidden_states)

    def _normalize(self, moments: torch.Tensor) -> torch.Tensor:
        latent_mean = moments.chunk(2, dim=1)[0]  # posterior mean (deterministic)
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latent_mean.device)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(latent_mean.device)
        return (latent_mean.float() - mean) / std

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode RGB image tensor [B, 3, H, W] in [0, 1] to normalized latents [B, 24, 1, H//16, W//16]."""
        if images.ndim == 4:
            pixels = images.unsqueeze(2)  # [B, 3, 1, H, W]
        else:
            pixels = images

        return self._normalize(self.encode_moments(pixels))

    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a clip [B, 3, T, H, W] in [0, 1] to normalized latents [B, 24, T_lat, H//16, W//16].

        Mirrors ComfyUI's MiniMaxH3VideoVAE.encode_temporal: the clip is cut into
        independent 17-frame chunks (the last one padded by repeating the final frame),
        each chunk encodes to 5 latent frames, and the trailing 3 latent frames are
        dropped. A T = 17k + 5 clip therefore yields 5k + 2 latent frames -- the grid H3's
        RoPE (FRAME_PER_TOKEN = 1, 4, 4, 4, 4) assumes. A single causal pass over the whole
        clip would give 1 + (T - 1) / 4 latents and misalign every frame after the first chunk.
        """
        if frames.shape[2] == 1:
            return self.encode_image(frames)
        if frames.shape[2] % VIDEO_CLIP_LENGTH:
            pad = VIDEO_CLIP_LENGTH - frames.shape[2] % VIDEO_CLIP_LENGTH
            frames = torch.cat([frames, frames[:, :, -1:].expand(-1, -1, pad, -1, -1)], dim=2)
        chunks = [
            self.encode_moments(frames[:, :, start : start + VIDEO_CLIP_LENGTH])
            for start in range(0, frames.shape[2], VIDEO_CLIP_LENGTH)
        ]
        moments = torch.cat(chunks, dim=2)[:, :, :-VIDEO_TOKEN_DROP]
        return self._normalize(moments)


# Temporal chunking of the reference VAE (comfy/ldm/minimax/vae.py: clip_length / token_drop).
VIDEO_CLIP_LENGTH = 17
VIDEO_TOKEN_DROP = 3


def align_frame_count(frames: int) -> int:
    """Largest valid H3 clip length <= frames: the 17k + 5 grid (5, 22, 39, 56, ...)."""
    if frames < 5:
        raise ValueError(f"MiniMax H3 clips need at least 5 frames, got {frames}")
    return 5 + ((frames - 5) // 17) * 17


def video_latent_frames(frames: int) -> int:
    """Latent frame count for a 17k + 5 frame clip (comfy_extras/nodes_minimax_h3.py)."""
    return 1 if frames == 1 else ((frames - 5) // 17) * 5 + 2


def _map_encoder_key(key: str) -> str | None:
    if not key.startswith(("encoder.", "quant_conv.")):
        return None
    if key.startswith("quant_conv."):
        return key
    rest = key[len("encoder.") :]
    if rest.startswith("down."):
        level, sub = rest[len("down.") :].split(".", 1)
        sub = sub.replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        sub = sub.replace("downsample.", "downsamplers.0.", 1)
        return f"down_blocks.{level}.{sub}"
    if rest.startswith("mid."):
        sub = rest[len("mid.") :].replace("block.", "resnets.", 1).replace("nin_shortcut.", "conv_shortcut.", 1)
        return f"mid_block.{sub}"
    return rest


def load_minimax_h3_video_vae(vae_path: str | Path, device: torch.device) -> MiniMaxH3VideoEncoderModel:
    """Load MiniMax H3 Video VAE encoder from safetensors checkpoint."""
    model = MiniMaxH3VideoEncoderModel()
    state_dict = load_file(str(vae_path))

    encoder_state_dict = {}
    for k, v in state_dict.items():
        mapped_key = _map_encoder_key(k)
        if mapped_key is not None:
            encoder_state_dict[mapped_key] = v

    info = model.load_state_dict(encoder_state_dict, strict=False)
    # latents_mean/latents_std are non-persistent buffers, so they are expected to be absent.
    unresolved = [key for key in info.missing_keys if key not in ("latents_mean", "latents_std")]
    if unresolved:
        raise ValueError(
            f"{Path(vae_path).name} did not supply {len(unresolved)} video VAE encoder tensor(s); "
            f"examples: {unresolved[:5]}"
        )
    if info.unexpected_keys:
        raise ValueError(
            f"{Path(vae_path).name} mapped {len(info.unexpected_keys)} tensor(s) the encoder does not "
            f"expect; examples: {info.unexpected_keys[:5]}"
        )

    model.to(device=device, dtype=torch.float16)
    model.eval()
    model.requires_grad_(False)
    return model
