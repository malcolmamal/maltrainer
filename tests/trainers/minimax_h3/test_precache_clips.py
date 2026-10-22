"""Pre-caching video clips: segment planning, soundtracks, muting, and the files that get skipped."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import train_minimax_h3_lora as h3
from maltrainer.trainers.minimax_h3 import audio as audio_parts

from .fakes import Fakes
from .media import write_clip
from .runs import cached
from .world import Workspace

LOGGER = "maltrainer.trainers.minimax_h3"  # its caching modules log under it


def _keys(trainer: h3.MiniMaxH3LoRATrainer) -> list[str]:
    return [
        Path(s.key.split("@")[0]).name + ("@" + s.key.split("@")[1] if "@" in s.key else "") for s in trainer.samples
    ]


def test_long_media_first_keeps_one_window_from_the_start_of_each_clip(workspace: Workspace) -> None:
    workspace.add_clip(frames=44, seconds=None)

    trainer = cached(workspace.config())

    assert [(s.frames, tuple(s.video.shape[2:3])) for s in trainer.samples] == [(22, (7,))]
    assert _keys(trainer) == ["clip.avi"]


def test_long_media_split_tiles_the_clip_and_keys_each_window_by_its_start_frame(workspace: Workspace) -> None:
    workspace.add_clip(frames=44, seconds=1.9)

    trainer = cached(workspace.config(dataset={"long_media": "split"}))

    assert _keys(trainer) == ["clip.avi@0", "clip.avi@22"]
    first, second = (s.audio for s in trainer.samples)
    assert first is not None and second is not None
    assert first.shape == second.shape == (1, 74, 32) and not torch.equal(first, second)


def test_a_clip_shorter_than_one_window_snaps_down_to_the_17k_plus_5_grid(workspace: Workspace) -> None:
    workspace.add_clip(frames=30, seconds=None)

    trainer = cached(workspace.config(dataset={"max_frames": 73}))

    assert [s.frames for s in trainer.samples] == [22]


def test_the_soundtrack_is_decoded_once_however_many_windows_the_clip_splits_into(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_clip(frames=44, seconds=1.9)
    reads: list[Path] = []
    real = audio_parts.read_audio

    def counting(path: Path) -> Any:
        reads.append(path)
        return real(path)

    monkeypatch.setattr(audio_parts, "read_audio", counting)

    cached(workspace.config(dataset={"long_media": "split"}))

    assert len(reads) == 1


def test_a_clip_with_no_audio_track_trains_as_a_muted_clip(workspace: Workspace) -> None:
    workspace.add_clip(seconds=None)

    trainer = cached(workspace.config())

    assert trainer.samples[0].audio is None
    assert trainer.run_metadata["dataset_composition"] == {
        "images": 0,
        "clips": 1,
        "clips_with_audio": 0,
        "audio_items": 0,
    }


def test_a_clip_whose_track_is_digital_silence_trains_as_a_muted_clip(workspace: Workspace) -> None:
    write_clip(workspace.data / "quiet.avi", 5, np.zeros((2, 4800), dtype=np.float32))

    trainer = cached(workspace.config())

    assert [s.audio for s in trainer.samples] == [None]


def test_a_stem_ending_in_mute_skips_the_soundtrack_without_decoding_it(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_clip("scene_mute", seconds=0.3)
    monkeypatch.setattr(audio_parts, "read_audio", lambda _path: pytest.fail("a muted clip's audio must not be read"))

    trainer = cached(workspace.config())

    assert trainer.samples[0].audio is None


def test_clip_audio_false_ignores_the_soundtrack(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_clip(seconds=0.3)

    trainer = cached(workspace.config(dataset={"clip_audio": False}))

    assert trainer.samples[0].audio is None and fakes.loads["audio"] == 0
