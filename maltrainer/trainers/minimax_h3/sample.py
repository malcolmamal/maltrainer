"""One cached H3 training item, and the dataset settings the caching reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from maltrainer.data.minimax_dataset import audio_latent_frames


@dataclass
class H3Sample:
    """One cached training item. Every kind carries a video latent and an audio row count."""

    key: str
    kind: str  # "image" | "clip" | "audio"
    caption: str
    frames: int  # 24 fps pixel frames the item spans (1 for a still)
    video: torch.Tensor  # [1, 24, T_lat, H/16, W/16] normalized, CPU
    audio: torch.Tensor | None = None  # [1, 2*A, 32] normalized rows, CPU; None = silence
    audio_only: bool = False  # video is a zeros placeholder; drop the video loss

    @property
    def audio_latents(self) -> int:
        return audio_latent_frames(self.frames)


@dataclass
class MediaSettings:
    """``dataset:`` as the caching reads it."""

    resolution: int
    clip_resolution: int
    aspect_mode: str
    max_frames: int
    long_media: str
    clip_audio: bool  # encode clip soundtracks (dataset.clip_audio, and only when audio trains)
    placeholder_px: int  # side of an audio-only item's zeros video
    max_segments: Any  # dataset.max_segments_per_file, read for long_media: split only

    @classmethod
    def from_config(cls, dataset_config: dict[str, Any], train_audio: bool) -> MediaSettings:
        resolution = int(dataset_config.get("resolution", 512))
        return cls(
            resolution=resolution,
            clip_resolution=int(dataset_config.get("clip_resolution", resolution)),
            aspect_mode=str(dataset_config.get("aspect_mode", "bucket")).lower(),
            max_frames=int(dataset_config.get("max_frames", 73)),
            long_media=str(dataset_config.get("long_media", "first")).lower(),
            clip_audio=bool(dataset_config.get("clip_audio", True)) and train_audio,
            placeholder_px=int(dataset_config.get("audio_placeholder_resolution", 128)),
            max_segments=dataset_config.get("max_segments_per_file", 16),
        )

    @property
    def frame_budget(self) -> int:
        """The frames read from one file: one clip's worth, or every segment's with long_media: split."""
        return self.max_frames if self.long_media != "split" else self.max_frames * int(self.max_segments)


def check_audio_rows(sample: H3Sample, rows: torch.Tensor) -> None:
    # The packed layout's audio grid is sized from the frame count, the rows from the
    # waveform. They agree by construction (hop-exact slicing); a silent mismatch would
    # train every audio row against a shifted position, so check rather than trust it.
    expected = sample.audio_latents * 2
    if rows.shape[1] != expected:
        raise ValueError(
            f"{sample.key}: audio encoded to {rows.shape[1]} rows, the {sample.frames}-frame layout packs {expected}"
        )
