"""Stage wrappers and the parameter-free shortcuts of the Qwen-Image-2.1 VAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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
        return x[:, :, ft - 1 :].squeeze(2)


class EncoderStage(nn.Module):
    def __init__(
        self,
        res_blocks: list[nn.Module],
        downsample: nn.Module | None = None,
        shortcut: nn.Module | None = None,
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
        res_blocks: list[nn.Module],
        upsample: nn.Module | None = None,
        shortcut: nn.Module | None = None,
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
