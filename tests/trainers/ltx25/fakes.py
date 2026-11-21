"""Stand-ins for the three heavy encoders the LTX-2.5 trainer loads: the video VAE, the audio VAE and the text encoder.
Everything else in a trainer run (the DiT, the LoRA network, the optimizer, the loop, the media decoding) is real."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional

from maltrainer.models.ltx25.config import AUDIO_BINS, AUDIO_CHANNELS
from maltrainer.training import session
from tests.maltrainer.models.ltx25.tiny_model import FEATURES


class FakeVideoVAE:
    """Pixels [1, 3, F, H, W] in [-1, 1] -> latents [1, 128, 1 + (F - 1) // 8, H / 32, W / 32]: a fixed projection."""

    def __init__(self) -> None:
        self.projection = torch.randn(128, 3, generator=torch.Generator().manual_seed(11))
        self.shapes: list[tuple[int, ...]] = []

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        self.shapes.append(tuple(pixels.shape))
        frames = 1 + (pixels.shape[2] - 1) // 8
        cells = functional.adaptive_avg_pool3d(pixels.float(), (frames, pixels.shape[3] // 32, pixels.shape[4] // 32))
        return torch.einsum("oc,bcfhw->bofhw", self.projection, cells)


class FakeAudioVAE:
    """A waveform [1, 2, L] -> latents [1, 8, T, 16] with T as the real causal VAE gives it for L samples at 16 kHz."""

    def __init__(self) -> None:
        self.lengths: list[int] = []

    def encode(self, waveform: torch.Tensor, rate: int) -> torch.Tensor:
        assert rate == 16000
        self.lengths.append(waveform.shape[-1])
        mel = waveform.shape[-1] // 160 + 1
        frames = ((mel - 1) // 2 + 1 - 1) // 2 + 1
        return waveform.float().mean() * torch.ones(1, AUDIO_CHANNELS, frames, AUDIO_BINS)


class FakeTextEncoder:
    """A caption -> [L, video + audio dims] features drawn from a generator seeded by the caption, L in 2..4."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, caption: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.prompts.append(caption)
        seed = zlib.crc32(caption.encode("utf-8"))
        features = torch.randn(2 + seed % 3, FEATURES, generator=torch.Generator().manual_seed(seed))
        return features.to(device, dtype)


@dataclass
class Fakes:
    video: FakeVideoVAE = field(default_factory=FakeVideoVAE)
    audio: FakeAudioVAE = field(default_factory=FakeAudioVAE)
    text: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    loads: list[str] = field(default_factory=list)


def install_fakes(monkeypatch: pytest.MonkeyPatch, trainer_module: object, cache_module: object) -> Fakes:
    """Replace the three loaders the trainer and its cache module imported with ones returning the fakes."""
    fakes = Fakes()

    def video(_path: str, _device: torch.device, _dtype: torch.dtype) -> FakeVideoVAE:
        fakes.loads.append("video")
        return fakes.video

    def audio(_path: str, _device: torch.device) -> FakeAudioVAE:
        fakes.loads.append("audio")
        return fakes.audio

    def text(_path: str, _device: torch.device, _dtype: torch.dtype) -> FakeTextEncoder:
        fakes.loads.append("text")
        return fakes.text

    monkeypatch.setattr(cache_module, "load_video_vae_encoder", video)
    monkeypatch.setattr(cache_module, "load_audio_vae_encoder", audio)
    monkeypatch.setattr(trainer_module, "load_ltx25_text_encoder", text)
    return fakes


def skip_housekeeping(monkeypatch: pytest.MonkeyPatch, commit: str = "abc1234") -> None:
    """The trainer's two slow environment calls, a full ``gc.collect()`` and ``git`` (twice), answered at once."""
    monkeypatch.setattr(session, "gc", SimpleNamespace(collect=lambda: 0))
    monkeypatch.setattr(session, "git_commit", lambda: commit)
