"""Pre-caching a mixed folder: what each kind of file becomes, and which stages load which model."""

from __future__ import annotations

import pytest
import torch

import train_minimax_h3_lora as h3
from maltrainer.data.media import read_audio
from maltrainer.data.minimax_dataset import load_image, slice_audio

from .fakes import Fakes
from .media import write_image
from .runs import cached
from .world import Workspace


def _mixed(ws: Workspace) -> h3.MiniMaxH3LoRATrainer:
    ws.add_images(2)
    ws.add_clip(frames=5, seconds=0.3)
    ws.add_voice(seconds=1.0, caption="a calm voice")
    return cached(ws.config())


def test_every_kind_becomes_a_sample_with_a_video_latent_and_an_audio_row_count(workspace: Workspace) -> None:
    trainer = _mixed(workspace)

    shapes = [(s.kind, s.frames, tuple(s.video.shape), s.audio_only, s.audio_latents) for s in trainer.samples]
    assert shapes == [
        ("clip", 5, (1, 24, 2, 2, 2), False, 8),
        ("image", 1, (1, 24, 1, 2, 2), False, 2),
        ("image", 1, (1, 24, 1, 2, 2), False, 2),
        ("audio", 22, (1, 24, 7, 2, 2), True, 37),
    ]


def test_a_still_is_encoded_as_an_image_and_carries_no_audio_rows(workspace: Workspace, fakes: Fakes) -> None:
    trainer = _mixed(workspace)

    image = next(s for s in trainer.samples if s.kind == "image")

    expected = fakes.video.encode_image(load_image(workspace.data / "stills" / "img0.png", 32, "bucket").unsqueeze(0))
    torch.testing.assert_close(image.video, expected)
    assert image.audio is None and image.key.endswith("img0.png")


def test_a_clip_soundtrack_and_a_recording_become_channel_major_audio_rows(workspace: Workspace, fakes: Fakes) -> None:
    trainer = _mixed(workspace)

    clip = next(s for s in trainer.samples if s.kind == "clip")
    voice = next(s for s in trainer.samples if s.kind == "audio")

    assert clip.audio is not None and clip.audio.shape == (1, 16, 32)
    assert voice.audio is not None and voice.audio.shape == (1, 74, 32)
    wav = read_audio(workspace.data / "voices" / "voice.wav")
    assert wav is not None
    expected = fakes.audio.encode(torch.from_numpy(slice_audio(wav, 0, 22)).unsqueeze(0))
    torch.testing.assert_close(voice.audio, h3.pack_audio_rows(expected))


def test_an_audio_item_gets_a_zeros_video_placeholder_sized_by_the_placeholder_resolution(
    workspace: Workspace,
) -> None:
    workspace.add_voice()

    trainer = cached(workspace.config(dataset={"audio_placeholder_resolution": 128}))

    placeholder = trainer.samples[0].video
    assert placeholder.shape == (1, 24, 7, 8, 8) and torch.count_nonzero(placeholder) == 0


def test_captions_come_from_sidecars_and_fall_back_to_the_trigger_and_default(workspace: Workspace) -> None:
    trainer = _mixed(workspace)

    assert {s.kind: s.caption for s in trainer.samples} == {
        "clip": "a photo of zed",
        "image": "a photo of zed",
        "audio": "a calm voice",
    }


def test_the_run_records_how_many_of_each_kind_it_cached(workspace: Workspace) -> None:
    trainer = _mixed(workspace)

    assert trainer.run_metadata["dataset_composition"] == {
        "images": 2,
        "clips": 1,
        "clips_with_audio": 1,
        "audio_items": 1,
    }


def test_the_video_audio_and_text_models_each_load_once_in_that_order(workspace: Workspace, fakes: Fakes) -> None:
    _mixed(workspace)

    assert fakes.loads == {"video": 1, "audio": 1, "text": 1, "transformer": 0}


def test_media_selects_the_kinds_to_scan_and_a_string_means_one_kind(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images(1)
    workspace.add_clip()
    workspace.add_voice()

    trainer = cached(workspace.config(dataset={"media": "image"}))

    assert [s.kind for s in trainer.samples] == ["image"]
    assert fakes.loads["audio"] == 0


def test_train_audio_false_drops_recordings_and_clip_soundtracks(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images(1)
    workspace.add_clip()
    workspace.add_voice()

    trainer = cached(workspace.config(train={"train_audio": False}))

    assert [(s.kind, s.audio) for s in trainer.samples] == [("clip", None), ("image", None)]
    assert fakes.loads["audio"] == 0


def test_a_recordings_only_folder_never_loads_the_video_vae(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_voice()

    cached(workspace.config())

    assert fakes.loads["video"] == 0 and fakes.loads["audio"] == 1


def test_bucket_keeps_the_aspect_ratio_and_stretch_squares_the_image(workspace: Workspace) -> None:
    write_image(workspace.data / "wide.png", "a wide one", size=(128, 64))

    bucket = cached(workspace.config(dataset={"resolution": 64}))
    stretch = cached(workspace.config(dataset={"resolution": 64, "aspect_mode": "Stretch"}))

    assert bucket.samples[0].video.shape[-2:] == (2, 4)
    assert stretch.samples[0].video.shape[-2:] == (4, 4)


@pytest.mark.parametrize(("resolution", "clip_resolution", "latent"), [(64, 32, (2, 2)), (32, 64, (4, 4))])
def test_clips_use_the_clip_resolution_and_stills_the_resolution(
    workspace: Workspace, resolution: int, clip_resolution: int, latent: tuple[int, int]
) -> None:
    workspace.add_clip(seconds=None)

    trainer = cached(workspace.config(dataset={"resolution": resolution, "clip_resolution": clip_resolution}))

    assert trainer.samples[0].video.shape[-2:] == latent
