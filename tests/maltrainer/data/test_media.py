from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from maltrainer.data.media import read_audio, read_video_frames
from maltrainer.data.minimax_dataset import AUDIO_SAMPLE_RATE


def _greys(frames: np.ndarray) -> list[int]:
    return [int(value) for value in frames[:, 0, 0, 0]]


def _tone(seconds: float, channels: int = 1, rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return np.tile(amplitude * np.sin(2 * np.pi * 440 * t), (channels, 1)).astype(np.float32)


def test_a_24_fps_clip_decodes_frame_for_frame(make_clip: Callable[..., Path]) -> None:
    path = make_clip("a.avi", frames=6, fps=24)

    frames = read_video_frames(path, 32, 100)

    assert frames.shape == (6, 32, 32, 3)
    assert frames.dtype == np.uint8
    assert _greys(frames) == [0, 20, 40, 60, 80, 100]


def test_a_12_fps_clip_holds_each_frame_for_two_ticks(make_clip: Callable[..., Path]) -> None:
    path = make_clip("a.avi", frames=6, fps=12)

    frames = read_video_frames(path, 32, 100)

    assert _greys(frames) == [0, 0, 20, 20, 40, 40, 60, 60, 80, 80, 100, 100]


def test_a_48_fps_clip_keeps_real_time_by_dropping_every_other_frame(make_clip: Callable[..., Path]) -> None:
    path = make_clip("a.avi", frames=12, fps=48)

    frames = read_video_frames(path, 32, 100)

    assert _greys(frames) == [0, 40, 80, 120, 160, 200]


@pytest.mark.parametrize(("fps", "max_frames", "expected"), [(24, 3, [0, 20, 40]), (12, 3, [0, 0, 20])])
def test_decoding_stops_at_max_frames(
    make_clip: Callable[..., Path], fps: int, max_frames: int, expected: list[int]
) -> None:
    path = make_clip("a.avi", frames=6, fps=fps)

    assert _greys(read_video_frames(path, 32, max_frames)) == expected


def test_frames_are_resized_to_the_bucket_of_the_clip(make_clip: Callable[..., Path]) -> None:
    path = make_clip("a.avi", frames=2, size=(48, 24))

    frames = read_video_frames(path, 64, 100)

    assert frames.shape == (2, 32, 64, 3)
    assert _greys(frames) == [0, 20]


def test_a_file_without_video_is_an_error(make_wav: Callable[..., Path]) -> None:
    path = make_wav("a.wav", _tone(0.1))

    with pytest.raises(ValueError, match="a.wav: no video stream"):
        read_video_frames(path, 32, 10)


def test_a_video_stream_without_frames_is_an_error(make_clip: Callable[..., Path]) -> None:
    path = make_clip("a.avi", frames=0)

    with pytest.raises(ValueError, match="a.avi: no decodable video frames"):
        read_video_frames(path, 32, 10)


def test_mono_audio_becomes_stereo_float_at_32_khz(make_wav: Callable[..., Path]) -> None:
    path = make_wav("a.wav", _tone(0.25), rate=16000)

    wav = read_audio(path)

    assert wav is not None
    assert wav.dtype == np.float32
    assert wav.shape == (2, AUDIO_SAMPLE_RATE // 4)
    assert np.array_equal(wav[0], wav[1])


def test_mono_audio_keeps_the_level_the_same_tone_has_in_stereo(make_wav: Callable[..., Path]) -> None:
    mono = read_audio(make_wav("mono.wav", _tone(0.25, channels=1)))
    stereo = read_audio(make_wav("stereo.wav", _tone(0.25, channels=2)))

    assert mono is not None
    assert stereo is not None
    np.testing.assert_allclose(mono, stereo, atol=1e-6)
    assert np.abs(mono).max() == pytest.approx(0.5, abs=0.01)


def test_stereo_audio_keeps_its_channels_apart(make_wav: Callable[..., Path]) -> None:
    samples = np.stack([np.full(8000, 0.25), np.full(8000, -0.25)]).astype(np.float32)
    path = make_wav("a.wav", samples, rate=32000)

    wav = read_audio(path)

    assert wav is not None
    assert wav.shape == (2, 8000)
    assert np.median(wav[0]) == pytest.approx(0.25, abs=1e-3)
    assert np.median(wav[1]) == pytest.approx(-0.25, abs=1e-3)


def test_samples_beyond_full_scale_are_clipped(make_wav: Callable[..., Path]) -> None:
    path = make_wav("a.wav", _tone(0.1, amplitude=1.5), rate=32000)

    wav = read_audio(path)

    assert wav is not None
    assert wav.max() == 1.0
    assert wav.min() == -1.0


def test_a_clips_soundtrack_is_read_alongside_its_video(make_clip: Callable[..., Path]) -> None:
    path = make_clip("a.avi", frames=6, audio=_tone(0.25, channels=2))

    wav = read_audio(path)

    assert wav is not None
    assert wav.shape == (2, AUDIO_SAMPLE_RATE // 4)
    assert np.abs(wav).max() > 0.4


@pytest.mark.parametrize("samples", [np.zeros((1, 4000), dtype=np.float32), np.zeros((1, 0), dtype=np.float32)])
def test_digital_silence_or_an_empty_track_has_no_sound_to_train(
    make_wav: Callable[..., Path], samples: np.ndarray
) -> None:
    assert read_audio(make_wav("a.wav", samples)) is None


def test_a_clip_without_an_audio_stream_has_no_sound(make_clip: Callable[..., Path]) -> None:
    assert read_audio(make_clip("a.avi", frames=2)) is None


def test_without_pyav_decoding_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "av", None)

    with pytest.raises(ImportError, match="pip install av"):
        read_audio(tmp_path / "a.wav")
