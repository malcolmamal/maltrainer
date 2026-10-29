"""LTX-2.5's media: stills and clips as pixels on the 8k + 1 frame grid, and the soundtrack that goes with each clip."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from maltrainer.data.media import read_audio, read_video_frames
from maltrainer.data.minimax_dataset import MediaSource, frames_tensor, load_image
from maltrainer.models.ltx25.audio_features import HOP, SAMPLE_RATE
from maltrainer.models.ltx25.rope import AUDIO_DOWNSAMPLE

LATENTS_PER_SECOND = SAMPLE_RATE / HOP / AUDIO_DOWNSAMPLE  # 25 audio latent frames per second
MIN_CLIP_FRAMES = 9  # a clip of fewer frames than one latent step beyond the first is not worth a video sample


@dataclass(frozen=True)
class MediaSettings:
    resolution: int = 512
    fps: float = 24.0
    clip_frames: int = 49
    audio: bool = True

    @property
    def max_frames(self) -> int:
        return grid_frames(self.clip_frames)


def grid_frames(frames: int) -> int:
    """The largest 8k + 1 frame count <= ``frames`` (the video VAE's grid), at least 1."""
    return max(1, 1 + (frames - 1) // 8 * 8)


def audio_latent_count(frames: int, fps: float) -> int:
    return round(frames / fps * LATENTS_PER_SECOND)


def audio_window(wav: np.ndarray, frames: int, fps: float) -> np.ndarray:
    """The first samples of the soundtrack that make the audio VAE emit exactly ``audio_latent_count`` latent frames.

    The causal audio VAE yields ``(mel frames - 1) // 4 + 1`` latents; ``(4 n - 3) * 160`` samples give n of them.
    Zero-padded at the end when the soundtrack is shorter than the clip."""
    want = (4 * audio_latent_count(frames, fps) - 3) * HOP
    window = wav[:, :want]
    return np.pad(window, ((0, 0), (0, want - window.shape[1]))) if window.shape[1] < want else window


def load_pixels(source: MediaSource, settings: MediaSettings) -> tuple[torch.Tensor, np.ndarray | None]:
    """The source as pixels [1, 3, F, H, W] in [-1, 1] (F on the 8k + 1 grid) and its soundtrack window, if any."""
    path: Path = source.path
    if source.kind == "image":
        image = load_image(path, settings.resolution)
        return (image.unsqueeze(0).unsqueeze(2) * 2.0 - 1.0), None
    frames = read_video_frames(path, settings.resolution, settings.max_frames, settings.fps)
    count = grid_frames(len(frames))
    if count < MIN_CLIP_FRAMES:
        raise ValueError(f"{path.name}: {len(frames)} frames at {settings.fps:g} fps, fewer than {MIN_CLIP_FRAMES}")
    pixels = frames_tensor(frames[:count]) * 2.0 - 1.0
    wav = read_audio(path, SAMPLE_RATE) if settings.audio else None
    return pixels, None if wav is None else audio_window(wav, count, settings.fps)
