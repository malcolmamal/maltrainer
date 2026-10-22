"""Pre-caching voice and sound recordings, and the text embeddings that follow every other stage."""

from __future__ import annotations

import logging

import pytest
import torch

from maltrainer.trainers.minimax_h3 import audio as audio_parts

from .fakes import Fakes
from .runs import cached
from .world import Workspace

LOGGER = "maltrainer.trainers.minimax_h3"  # its caching modules log under it


def test_a_long_recording_keeps_one_window_in_first_mode_and_tiles_in_split_mode(workspace: Workspace) -> None:
    workspace.add_voice(seconds=2.0)

    first = cached(workspace.config())
    split = cached(workspace.config(dataset={"long_media": "split"}))

    assert [(s.frames, s.key.endswith("voice.wav")) for s in first.samples] == [(22, True)]
    assert [s.key.rsplit("@", 1)[-1] for s in split.samples] == ["0", "22"]
    first_rows, second_rows = (s.audio for s in split.samples)
    assert first_rows is not None and second_rows is not None and not torch.equal(first_rows, second_rows)


def test_audio_only_samples_drop_the_video_loss_and_keep_their_caption(workspace: Workspace) -> None:
    workspace.add_voice(caption="she whispers")

    sample = cached(workspace.config()).samples[0]

    assert (sample.kind, sample.audio_only, sample.caption) == ("audio", True, "she whispers")


def test_recordings_need_the_audio_vae_path(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_voice()
    config = workspace.config()
    del config["model"]["audio_vae_path"]

    with pytest.raises(ValueError, match=r"model\.audio_vae_path is required .* Set train\.train_audio: false"):
        cached(config)

    assert fakes.loads["audio"] == 0


def test_clip_soundtracks_need_the_audio_vae_path_too(workspace: Workspace) -> None:
    workspace.add_clip(seconds=0.3)
    config = workspace.config()
    config["model"]["audio_vae_path"] = ""

    with pytest.raises(ValueError, match=r"model\.audio_vae_path is required"):
        cached(config)


def test_a_silent_recording_is_skipped_with_a_warning(workspace: Workspace, caplog: pytest.LogCaptureFixture) -> None:
    workspace.add_silence("hush")
    workspace.add_voice("talk")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        trainer = cached(workspace.config())

    assert [s.key.endswith("talk.wav") for s in trainer.samples] == [True]
    assert "Skipping hush.wav: no decodable sound (or digital silence)" in caplog.text


def test_a_recording_shorter_than_a_second_is_skipped_with_a_warning(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_voice("blip", seconds=0.4)
    workspace.add_voice("talk")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        trainer = cached(workspace.config())

    assert len(trainer.samples) == 1
    assert "Skipping blip.wav: shorter than 0.92 s" in caplog.text


def test_a_recording_that_encodes_to_the_wrong_row_count_is_skipped(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_voice()
    encode = fakes.audio.encode
    monkeypatch.setattr(fakes.audio, "encode", lambda wav: encode(wav)[..., :-1])

    with caplog.at_level(logging.WARNING, logger=LOGGER), pytest.raises(ValueError, match="Every dataset item"):
        cached(workspace.config())

    assert "voice.wav: audio encoded to 72 rows, the 22-frame layout packs 74" in caplog.text


def test_a_missing_dependency_while_reading_a_recording_is_not_swallowed(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_voice()

    def no_av(_path: object) -> None:
        raise ImportError("Video and audio training items need PyAV")

    monkeypatch.setattr(audio_parts, "read_audio", no_av)

    with pytest.raises(ImportError, match="need PyAV"):
        cached(workspace.config())


def test_a_folder_of_unusable_media_is_refused(workspace: Workspace) -> None:
    workspace.add_silence()

    with pytest.raises(ValueError, match=r"Every dataset item was skipped; see the warnings above\."):
        cached(workspace.config())


def test_an_empty_folder_is_refused_naming_the_kinds_and_the_path(workspace: Workspace) -> None:
    workspace.data.mkdir()

    with pytest.raises(ValueError, match=r"No training media \(image, clip, audio\) found under .*data"):
        cached(workspace.config())


def test_a_folder_with_only_recordings_is_refused_when_train_audio_is_off(workspace: Workspace) -> None:
    workspace.add_voice()

    with pytest.raises(ValueError, match=r"No training media \(image, clip\) found under"):
        cached(workspace.config(train={"train_audio": False}))


def test_an_unknown_media_kind_is_refused(workspace: Workspace) -> None:
    workspace.add_images(1)

    with pytest.raises(ValueError, match=r"Unknown dataset media kind\(s\) \['gif'\]"):
        cached(workspace.config(dataset={"media": ["image", "gif"]}))
