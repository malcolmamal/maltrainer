"""A clip maps onto the 24 fps clock the same way whatever container it is in. MKV keeps
timestamps in whole milliseconds, so without snapping them back onto the frame grid a 24 fps MKV
lost frames 1 and 4, and a 12 fps one decoded 13 frames instead of 12."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from maltrainer.data.media import _on_grid, read_video_frames


def _greys(frames: np.ndarray) -> list[int]:
    return [int(frame[0, 0, 0]) for frame in frames]


@pytest.mark.parametrize(
    ("fps", "frames", "expected"),
    [
        (24, 6, [0, 20, 40, 60, 80, 100]),
        (12, 6, [0, 0, 20, 20, 40, 40, 60, 60, 80, 80, 100, 100]),
        (30, 10, [0, 20, 40, 60, 100, 120, 140, 160]),
        (48, 12, [0, 40, 80, 120, 160, 200]),
    ],
)
def test_an_mkv_decodes_like_the_same_clip_in_avi(
    make_clip: Callable[..., Path], fps: int, frames: int, expected: list[int]
) -> None:
    avi = read_video_frames(make_clip("a.avi", frames=frames, fps=fps), 32, 100)
    mkv = read_video_frames(make_clip("a.mkv", frames=frames, fps=fps), 32, 100)

    assert _greys(avi) == expected
    assert _greys(mkv) == expected


@pytest.mark.parametrize(
    ("time", "expected"),
    [(0.042, 1 / 24), (0.041, 1 / 24), (0.0835, 2 / 24), (0.0, 0.0), (0.05, 0.05), (0.030, 0.030)],
)
def test_a_time_within_the_snap_of_the_grid_moves_onto_it(time: float, expected: float) -> None:
    assert _on_grid(time, 1 / 24, 0.001) == pytest.approx(expected, abs=1e-12)
