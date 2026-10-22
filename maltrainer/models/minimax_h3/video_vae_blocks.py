"""Down-sampling and middle blocks of the MiniMax H3 video VAE."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from maltrainer.models.minimax_h3.video_vae_layers import MiniMaxH3VideoCausalConv3d, MiniMaxH3VideoResnetBlock3d


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
        return cast(torch.Tensor, self.conv(hidden_states))


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
