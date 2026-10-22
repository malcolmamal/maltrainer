"""
MiniMax H3 Video VAE wrapper and encoding logic.

The layers live in ``video_vae_layers`` and ``video_vae_blocks``, the latent statistics in ``video_vae_constants``,
the frame-count helpers in ``video_frames`` and the checkpoint key map in ``video_vae_keys``; every name is
re-exported here.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from maltrainer.models.minimax_h3.video_frames import (
    VIDEO_CLIP_LENGTH,
    VIDEO_TOKEN_DROP,
    align_frame_count,
    video_latent_frames,
)
from maltrainer.models.minimax_h3.video_vae_blocks import (
    MiniMaxH3VideoDownBlock3d,
    MiniMaxH3VideoDownsample3d,
    MiniMaxH3VideoMidBlock3d,
)
from maltrainer.models.minimax_h3.video_vae_constants import DEFAULT_LATENTS_MEAN, DEFAULT_LATENTS_STD
from maltrainer.models.minimax_h3.video_vae_keys import map_encoder_key as _map_encoder_key
from maltrainer.models.minimax_h3.video_vae_layers import (
    MiniMaxH3VideoCausalConv3d,
    MiniMaxH3VideoGroupNorm,
    MiniMaxH3VideoResnetBlock3d,
)

__all__ = [
    "DEFAULT_LATENTS_MEAN",
    "DEFAULT_LATENTS_STD",
    "VIDEO_CLIP_LENGTH",
    "VIDEO_TOKEN_DROP",
    "MiniMaxH3VideoCausalConv3d",
    "MiniMaxH3VideoDownBlock3d",
    "MiniMaxH3VideoDownsample3d",
    "MiniMaxH3VideoEncoderModel",
    "MiniMaxH3VideoGroupNorm",
    "MiniMaxH3VideoMidBlock3d",
    "MiniMaxH3VideoResnetBlock3d",
    "_map_encoder_key",
    "align_frame_count",
    "load_minimax_h3_video_vae",
    "video_latent_frames",
]


class MiniMaxH3VideoEncoderModel(nn.Module):
    """MiniMax H3 Video VAE Encoder."""

    latents_mean: torch.Tensor
    latents_std: torch.Tensor

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
        latents_mean: tuple[float, ...] = DEFAULT_LATENTS_MEAN,
        latents_std: tuple[float, ...] = DEFAULT_LATENTS_STD,
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

        block_in_channels = (block_out_channels[0], *tuple(block_out_channels[:-1]))
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
        hidden_states = ((pixels.float() - pixel_mean) / pixel_std).to(dtype=self.conv_in.weight.dtype)
        hidden_states = self.conv_in(hidden_states)
        for block in self.down_blocks:
            hidden_states = block(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        hidden_states = self.conv_out(hidden_states)
        return cast(torch.Tensor, self.quant_conv(hidden_states))

    def _normalize(self, moments: torch.Tensor) -> torch.Tensor:
        latent_mean = moments.chunk(2, dim=1)[0]  # posterior mean (deterministic)
        mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latent_mean.device)
        std = self.latents_std.view(1, -1, 1, 1, 1).to(latent_mean.device)
        return (latent_mean.float() - mean) / std

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode RGB image tensor [B, 3, H, W] in [0, 1] to normalized latents [B, 24, 1, H//16, W//16]."""
        pixels = images.unsqueeze(2) if images.ndim == 4 else images  # [B, 3, 1, H, W]
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
