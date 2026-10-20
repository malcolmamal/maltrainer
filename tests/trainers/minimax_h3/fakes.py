"""Stand-ins for what an H3 run loads from giant checkpoints. The VAEs and the transformer are the repo's own
tiny models; the text encoder is deterministic. Everything between them is the trainer's real code."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from maltrainer.models.minimax_h3.audio_vae import MiniMaxH3AudioEncoder
from maltrainer.models.minimax_h3.model import MiniMaxH3TransformerConfig
from maltrainer.models.minimax_h3.video_vae import MiniMaxH3VideoEncoderModel
from maltrainer.trainers.minimax_h3 import audio as audio_parts
from maltrainer.trainers.minimax_h3 import data, model, visual
from maltrainer.training import session
from tests.maltrainer.models.minimax_h3.h3_audio import tiny_audio_encoder
from tests.maltrainer.models.minimax_h3.h3_tiny import TEXT_DIM, tiny_config, tiny_transformer

AUDIO_CHANNELS = 32  # the trainer packs 32-channel audio rows
VIDEO_CHANNELS = 24  # ... and its audio-only placeholder is a 24-channel video latent


def tiny_video_vae(seed: int = 0) -> MiniMaxH3VideoEncoderModel:
    """The repo's video encoder at toy width: 16x spatial and 4x temporal downsampling, 24 latent channels."""
    torch.manual_seed(seed)
    model = MiniMaxH3VideoEncoderModel(
        latent_channels=VIDEO_CHANNELS,
        block_out_channels=(4, 4, 8, 8),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2, 2, 2),
        temporal_downsample_factors=(1, 2, 2, 1),
        norm_num_groups=2,
    )
    return model.eval().requires_grad_(False)


class FakeTextEncoder:
    """A caption -> [1, L, 6] embedding from a generator seeded by the caption; L is its word count (>= 1)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode_prompts(self, prompts: list[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        caption = prompts[0]
        self.prompts.append(caption)
        seed = zlib.crc32(caption.encode("utf-8"))
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(1, max(1, len(caption.split())), TEXT_DIM, generator=generator).to(device, dtype)


@dataclass
class Fakes:
    video: MiniMaxH3VideoEncoderModel
    audio: MiniMaxH3AudioEncoder
    text: FakeTextEncoder = field(default_factory=FakeTextEncoder)
    adaln_table: int | None = None  # rows of a pruned AdaLN table; None = the full time embedder
    loads: dict[str, int] = field(default_factory=lambda: {"video": 0, "audio": 0, "text": 0, "transformer": 0})
    configs: list[MiniMaxH3TransformerConfig] = field(default_factory=list)
    text_options: list[tuple[str | None, torch.device, torch.dtype]] = field(default_factory=list)

    def reference_state(self) -> dict[str, torch.Tensor]:
        """The base checkpoint's tensors as the trainer first loads them (a fresh copy each call)."""
        options: dict[str, Any] = {"audio_in_channels": AUDIO_CHANNELS, "in_channels": VIDEO_CHANNELS}
        if self.adaln_table is not None:
            options.update(adaln_t_table_size=self.adaln_table)
        with torch.random.fork_rng():  # fixed weights, without touching the RNG the trainer seeded
            model = tiny_transformer(**options)
        if self.adaln_table is not None:
            model.get_buffer("adaln_t_table").copy_(torch.randn(self.adaln_table, 10, generator=_gen()))
        return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _gen() -> torch.Generator:
    return torch.Generator().manual_seed(5)


def install_fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    """Replace the loaders the trainer's parts import, and the full-size transformer config, with tiny ones."""
    fakes = Fakes(video=tiny_video_vae(), audio=tiny_audio_encoder(monkeypatch))

    def video(_path: str, device: torch.device) -> MiniMaxH3VideoEncoderModel:
        fakes.loads["video"] += 1
        return fakes.video.to(device)

    def audio(_path: str, device: torch.device) -> MiniMaxH3AudioEncoder:
        fakes.loads["audio"] += 1
        return fakes.audio.to(device)

    def text(_path: str, tokenizer_path: str | None, device: torch.device, dtype: torch.dtype) -> FakeTextEncoder:
        fakes.loads["text"] += 1
        fakes.text_options.append((tokenizer_path, device, dtype))
        return fakes.text

    def state(_path: str, device: torch.device) -> tuple[dict[str, torch.Tensor], int]:
        fakes.loads["transformer"] += 1
        return {key: value.to(device) for key, value in fakes.reference_state().items()}, 0

    def config(**options: Any) -> MiniMaxH3TransformerConfig:
        fakes.configs.append(tiny_config(audio_in_channels=AUDIO_CHANNELS, in_channels=VIDEO_CHANNELS, **options))
        return fakes.configs[-1]

    monkeypatch.setattr(visual, "load_minimax_h3_video_vae", video)
    monkeypatch.setattr(audio_parts, "load_minimax_h3_audio_encoder", audio)
    monkeypatch.setattr(data, "load_minimax_h3_text_encoder", text)
    monkeypatch.setattr(model, "load_comfy_int8_convrot_state_dict", state)
    monkeypatch.setattr(model, "MiniMaxH3TransformerConfig", config)
    return fakes


def skip_housekeeping(monkeypatch: pytest.MonkeyPatch, commit: str = "abc1234") -> None:
    """The trainer's two slow environment calls, a full ``gc.collect()`` and ``git``, answered at once."""
    monkeypatch.setattr(session, "gc", SimpleNamespace(collect=lambda: 0))
    monkeypatch.setattr(session, "git_commit", lambda: commit)
