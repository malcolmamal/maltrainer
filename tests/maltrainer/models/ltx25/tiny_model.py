"""An LTX-2.5 DiT small enough for CPU tests: 2 blocks of 2 heads, the video stream 96 wide and the audio stream 48."""

from __future__ import annotations

import torch

from maltrainer.models.ltx25.config import AUDIO_BINS, AUDIO_CHANNELS, ConnectorConfig, LTX25Config
from maltrainer.models.ltx25.model import LTX25Transformer

TINY = LTX25Config(
    num_layers=2,
    num_heads=2,
    head_dim=48,
    audio_num_heads=2,
    audio_head_dim=24,
    cross_attention_dim=96,
    audio_cross_attention_dim=48,
    max_pos=(20, 64, 64),
    video_connector=ConnectorConfig(num_heads=2, head_dim=48, num_layers=1, registers=128, gated=True),
    audio_connector=ConnectorConfig(num_heads=2, head_dim=24, num_layers=1, registers=128, gated=True),
)
FEATURES = TINY.cross_attention_dim + TINY.audio_cross_attention_dim


def randomise(model: torch.nn.Module, seed: int = 0) -> None:
    """Small random weights everywhere (the real zero-initialised gates and tables would hide most of the network)."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)


def tiny_transformer(config: LTX25Config = TINY, with_audio: bool = True, seed: int = 0) -> LTX25Transformer:
    model = LTX25Transformer(config, with_audio)
    randomise(model, seed)
    model.compute_dtype = torch.float32
    return model.eval()


def tiny_inputs(
    frames: int = 3, height: int = 2, width: int = 4, audio_frames: int = 5, length: int = 7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(video, sigma, features, audio, audio_sigma)."""
    generator = torch.Generator().manual_seed(3)
    video = torch.randn(1, 128, frames, height, width, generator=generator)
    audio = torch.randn(1, AUDIO_CHANNELS, audio_frames, AUDIO_BINS, generator=generator)
    features = torch.randn(1, length, FEATURES, generator=generator)
    return video, torch.tensor([0.37]), features, audio, torch.tensor([0.61])
