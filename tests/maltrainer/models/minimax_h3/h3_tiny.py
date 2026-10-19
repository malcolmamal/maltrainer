from __future__ import annotations

from dataclasses import dataclass

import torch

from maltrainer.models.minimax_h3.model import (
    MiniMaxH3TokenTag,
    MiniMaxH3Transformer,
    MiniMaxH3TransformerConfig,
)

HIDDEN = 16
TEXT_DIM = 6
VIDEO_PATCH = 8
AUDIO_CHANNELS = 3


def tiny_config(**overrides: object) -> MiniMaxH3TransformerConfig:
    options: dict = {
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "hidden_size": HIDDEN,
        "num_layers": 3,
        "num_refiner_layers": 1,
        "ffn_dim": 8,
        "in_channels": 2,
        "audio_in_channels": AUDIO_CHANNELS,
        "text_dim": TEXT_DIM,
        "freq_dim": 8,
        "time_embed_hidden_dim": 12,
        "time_embed_dim": 10,
        "rope_freq_dim": 1,
    }
    options.update(overrides)
    return MiniMaxH3TransformerConfig(**options)


def tiny_transformer(seed: int = 0, **overrides: object) -> MiniMaxH3Transformer:
    torch.manual_seed(seed)
    return MiniMaxH3Transformer(tiny_config(**overrides)).eval()


@dataclass
class Packed:
    """A [text | audio | video] sequence: index sets, tags, timestep rows and per-token positions."""

    text: int
    audio: int
    video: int

    @property
    def length(self) -> int:
        return self.text + self.audio + self.video

    def indices(self) -> dict[str, torch.Tensor]:
        text = torch.arange(0, self.text)
        audio = torch.arange(self.text, self.text + self.audio)
        video = torch.arange(self.text + self.audio, self.length)
        tags = torch.zeros(self.length, dtype=torch.long)
        tags[text], tags[audio], tags[video] = MiniMaxH3TokenTag.TEXT, MiniMaxH3TokenTag.AUDIO, MiniMaxH3TokenTag.VIDEO
        timestep_indices = torch.zeros(self.length, dtype=torch.long)
        timestep_indices[audio] = 1
        positions = torch.zeros(self.length, 3)
        positions[:, 0] = torch.arange(self.length)
        return {
            "text_indices": text,
            "audio_indices": audio,
            "video_indices": video,
            "token_tags": tags,
            "timestep_indices": timestep_indices,
            "position_ids": positions,
        }

    def inputs(self, batch: int = 1, seed: int = 0) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return {
            "video_hidden_states": torch.randn(batch, self.video, VIDEO_PATCH, generator=generator),
            "audio_hidden_states": torch.randn(batch, self.audio, AUDIO_CHANNELS, generator=generator),
            "encoder_hidden_states": torch.randn(batch, self.text, TEXT_DIM, generator=generator),
            "timestep": torch.tensor([0.25, 0.75]),
            **self.indices(),
        }
