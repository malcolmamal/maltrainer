"""Step 1 of a run: the dataset scan, the video, audio and text caches and what happens to bad files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from train_ltx25_lora import LTX25LoRATrainer

from .fakes import Fakes
from .runs import run_lora
from .world import Workspace


def _cached(ws: Workspace, **sections: Any) -> LTX25LoRATrainer:
    trainer = LTX25LoRATrainer(ws.config(**sections))
    trainer.precache_dataset()
    return trainer


def test_every_image_and_clip_gets_a_latent_and_every_distinct_caption_one_feature_matrix(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images(2)
    workspace.add_clip("clip0", frames=9)
    workspace.add_clip("clip1", frames=17, caption="zed runs")

    trainer = _cached(workspace, dataset={"clip_frames": 17})

    names = [Path(key).name for key, _ in trainer.samples]
    assert names == ["clip0.avi", "clip1.avi", "img0.png", "img1.png"]
    shapes = {Path(key).name: tuple(trainer.latent_cache[key].shape) for key, _ in trainer.samples}
    assert shapes == {
        "clip0.avi": (128, 2, 1, 1),
        "clip1.avi": (128, 3, 1, 1),
        "img0.png": (128, 1, 1, 1),
        "img1.png": (128, 1, 1, 1),
    }
    assert sorted(trainer.text_cache) == ["a photo of zed", "zed runs", "zed walks"]
    assert fakes.loads == ["video", "audio", "text"]


def test_a_clip_with_sound_gets_audio_latents_cut_to_its_length_and_a_still_or_silent_clip_gets_none(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images(1)
    workspace.add_clip("loud", frames=9, seconds=0.5)
    workspace.add_clip("mute", frames=9, seconds=None)

    trainer = _cached(workspace)

    audio = {Path(key).name for key in trainer.audio_cache}
    assert audio == {"loud.avi"} and trainer.with_audio
    # 9 frames at 24 fps = 0.375 s = 9 audio latents (25 a second, rounded); (4 * 9 - 3) * 160 samples make exactly 9
    assert fakes.audio.lengths == [(4 * 9 - 3) * 160]
    assert next(iter(trainer.audio_cache.values())).shape == (8, 9, 16)


def test_a_short_soundtrack_is_padded_with_silence_to_the_clips_length(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_clip("short", frames=17, seconds=0.2)

    trainer = _cached(workspace, dataset={"clip_frames": 17})

    assert fakes.audio.lengths == [(4 * round(17 / 24 * 25) - 3) * 160]
    assert next(iter(trainer.audio_cache.values())).shape[1] == round(17 / 24 * 25)


def test_with_audio_switched_off_the_audio_vae_never_loads_and_the_model_will_be_video_only(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_clip("loud", seconds=0.5)

    trainer = _cached(workspace, dataset={"audio": False})

    assert trainer.audio_cache == {} and not trainer.with_audio and "audio" not in fakes.loads


def test_without_any_soundtrack_nothing_needs_the_audio_vae(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images(2)

    trainer = _cached(workspace)

    assert not trainer.with_audio and fakes.loads == ["video", "text"]


def test_sound_without_an_audio_vae_path_is_refused_with_the_way_out(workspace: Workspace) -> None:
    workspace.add_clip("loud", seconds=0.5)
    config = workspace.config()
    del config["model"]["audio_vae_path"]

    with pytest.raises(ValueError, match=r"model\.audio_vae_path is required.*dataset\.audio: false"):
        LTX25LoRATrainer(config).precache_dataset()


def test_a_clip_with_fewer_than_nine_frames_is_skipped_with_a_warning(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)
    workspace.add_clip("tiny", frames=5)

    with caplog.at_level(logging.WARNING):
        trainer = _cached(workspace)

    assert [Path(key).name for key, _ in trainer.samples] == ["img0.png"]
    assert "Skipping tiny.avi" in caplog.text and "fewer than 9" in caplog.text


def test_the_clip_length_is_cut_to_the_configured_frames_on_the_eight_k_plus_one_grid(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_clip("long", frames=30, seconds=None)

    _cached(workspace, dataset={"clip_frames": 20})  # 20 -> 17

    assert fakes.video.shapes == [(1, 3, 17, 32, 32)]


def test_caption_dropout_adds_the_empty_prompt_to_the_cache(workspace: Workspace) -> None:
    workspace.add_images(1)

    plain = _cached(workspace)
    dropping = _cached(workspace, train={"caption_dropout": 0.1})

    assert sorted(plain.text_cache) == ["a photo of zed"] and sorted(dropping.text_cache) == ["", "a photo of zed"]


def test_an_unreadable_file_is_skipped_with_a_warning_and_the_rest_train(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    (workspace.data / "broken.png").write_bytes(b"not an image")

    with caplog.at_level(logging.WARNING):
        trainer = run_lora(workspace.config())

    assert len(trainer.samples) == 2 and "Skipping broken.png:" in caplog.text
    assert (workspace.out / "ltx25.safetensors").exists()


def test_a_folder_without_media_is_refused_before_any_model_loads(workspace: Workspace, fakes: Fakes) -> None:
    with pytest.raises(ValueError, match=r"No images or videos found under .*data"):
        run_lora(workspace.config())

    assert fakes.loads == []


def test_a_mismatched_audio_latent_count_drops_that_soundtrack_with_a_warning(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_clip("loud", seconds=0.5)
    monkeypatch.setattr(fakes.audio, "encode", lambda *_: torch.zeros(1, 8, 3, 16))

    with caplog.at_level(logging.WARNING):
        trainer = _cached(workspace)

    assert trainer.audio_cache == {} and "3 audio latents, expected 9" in caplog.text


def test_the_cache_logs_how_many_samples_have_sound(workspace: Workspace, caplog: pytest.LogCaptureFixture) -> None:
    workspace.add_images(1)
    workspace.add_clip("loud", seconds=0.5)

    with caplog.at_level(logging.INFO):
        _cached(workspace)

    assert "Pre-cached 2 samples (1 with sound) and 2 prompts." in caplog.text


def test_a_clip_the_decoder_rejects_is_skipped_but_a_missing_pyav_stops_the_run(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)
    (workspace.data / "garbage.avi").write_bytes(b"not a video")

    with caplog.at_level(logging.WARNING):
        trainer = _cached(workspace)

    assert [Path(key).name for key, _ in trainer.samples] == ["img0.png"] and "Skipping garbage.avi" in caplog.text

    def no_av(*_: object) -> None:
        raise ImportError("Video and audio training items need PyAV: pip install av")

    monkeypatch.setattr("maltrainer.trainers.ltx25.cache.load_pixels", no_av)
    with pytest.raises(ImportError, match="PyAV"):
        _cached(workspace)
