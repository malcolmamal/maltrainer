from __future__ import annotations

import numpy as np
import pytest

from maltrainer.data.minimax_dataset import MIN_AUDIO_FRAMES, MIN_CLIP_FRAMES, plan_segments, slice_audio


@pytest.mark.parametrize(
    ("total", "max_frames", "mode", "min_frames", "expected"),
    [
        (100, 73, "first", MIN_CLIP_FRAMES, [(0, 73)]),
        (30, 73, "first", MIN_CLIP_FRAMES, [(0, 22)]),
        (8, 73, "first", MIN_CLIP_FRAMES, [(0, 5)]),
        (4, 73, "first", MIN_CLIP_FRAMES, []),
        (10, 73, "first", MIN_AUDIO_FRAMES, []),
        (60, 80, "first", MIN_CLIP_FRAMES, [(0, 56)]),
        (100, 73, "split", MIN_CLIP_FRAMES, [(0, 73), (73, 22)]),
        (146, 73, "split", MIN_CLIP_FRAMES, [(0, 73), (73, 73)]),
        (90, 73, "split", MIN_CLIP_FRAMES, [(0, 73)]),
        (30, 73, "split", MIN_CLIP_FRAMES, [(0, 22)]),
        (30, 22, "split", MIN_CLIP_FRAMES, [(0, 22)]),
        (66, 22, "split", MIN_AUDIO_FRAMES, [(0, 22), (22, 22), (44, 22)]),
    ],
)
def test_segments_tile_the_source_on_the_clip_grid(
    total: int, max_frames: int, mode: str, min_frames: int, expected: list[tuple[int, int]]
) -> None:
    assert plan_segments(total, max_frames, mode, min_frames) == expected


@pytest.mark.parametrize(("max_frames", "min_frames"), [(4, MIN_CLIP_FRAMES), (21, MIN_AUDIO_FRAMES)])
def test_a_max_length_below_one_minimum_window_is_an_error(max_frames: int, min_frames: int) -> None:
    with pytest.raises(ValueError, match="max_frames must allow at least"):
        plan_segments(100, max_frames, "first", min_frames)


def test_an_audio_window_starts_at_the_frames_time_and_is_hop_exact() -> None:
    wav = np.tile(np.arange(64000, dtype=np.float32), (2, 1))

    window = slice_audio(wav, 24, 5)

    assert window.shape == (2, 6400)
    assert window[0, 0] == 32000
    assert window[1, -1] == 32000 + 6399


def test_a_short_recording_is_zero_padded_to_the_window() -> None:
    wav = np.ones((2, 1000), dtype=np.float32)

    window = slice_audio(wav, 0, 22)

    assert window.shape == (2, 29600)
    assert window[:, :1000].sum() == 2000
    assert not window[:, 1000:].any()


def test_a_window_past_the_end_is_silence() -> None:
    window = slice_audio(np.ones((2, 100), dtype=np.float32), 24, 5)

    assert window.shape == (2, 6400)
    assert not window.any()
