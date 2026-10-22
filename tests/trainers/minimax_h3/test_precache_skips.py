"""Clips that cannot be cached: too short, undecodable, mis-encoded or needing a missing dependency."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

import train_minimax_h3_lora as h3
from maltrainer.trainers.minimax_h3 import visual

from .fakes import Fakes
from .runs import cached
from .world import Workspace

LOGGER = "maltrainer.trainers.minimax_h3"  # its caching modules log under it


def _keys(trainer: h3.MiniMaxH3LoRATrainer) -> list[str]:
    return [
        Path(s.key.split("@")[0]).name + ("@" + s.key.split("@")[1] if "@" in s.key else "") for s in trainer.samples
    ]


def test_a_clip_with_fewer_than_five_frames_is_skipped_with_a_warning(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_clip("blip", frames=3, seconds=None)
    workspace.add_clip("ok", frames=5, seconds=None)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        trainer = cached(workspace.config())

    assert _keys(trainer) == ["ok.avi"]
    assert "Skipping blip.avi: 3 frames is shorter than one H3 clip" in caplog.text


def test_an_undecodable_file_is_skipped_and_the_rest_of_the_folder_still_trains(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    (workspace.data / "clips").mkdir(parents=True)
    (workspace.data / "clips" / "bad.avi").write_bytes(b"not a video at all")
    workspace.add_clip("ok", seconds=None)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        trainer = cached(workspace.config())

    assert _keys(trainer) == ["ok.avi"]
    assert "Skipping bad.avi:" in caplog.text


def test_a_missing_dependency_is_not_swallowed_as_a_skipped_file(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_clip(seconds=None)

    def no_av(*_args: object) -> None:
        raise ImportError("Video and audio training items need PyAV")

    monkeypatch.setattr(visual, "read_video_frames", no_av)

    with pytest.raises(ImportError, match="need PyAV"):
        cached(workspace.config())


def test_a_clip_that_encodes_to_the_wrong_latent_count_is_skipped(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_clip(seconds=None)
    encode = fakes.video.encode_video
    monkeypatch.setattr(fakes.video, "encode_video", lambda frames: encode(frames)[:, :, :-1])

    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(ValueError, match="Every dataset item"):
        cached(workspace.config())

    assert "clip.avi: 5 frames encoded to 1 latent frames, expected 2" in caplog.text


def test_a_soundtrack_that_encodes_to_the_wrong_row_count_stops_the_run(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_clip(seconds=0.3)
    encode = fakes.audio.encode
    monkeypatch.setattr(fakes.audio, "encode", lambda wav: torch.cat([encode(wav), encode(wav)[..., -1:]], dim=-1))

    with pytest.raises(ValueError, match=r"clip\.avi: audio encoded to 18 rows, the 5-frame layout packs 16"):
        cached(workspace.config())
