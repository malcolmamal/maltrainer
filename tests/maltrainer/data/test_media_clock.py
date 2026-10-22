"""The timing arithmetic of the video reader, on fake PyAV frames (no files, no codecs)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from maltrainer.data import media
from maltrainer.data.media import _frame_timing, _on_grid, read_video_frames


def _frame(grey: int, time: float | None) -> Any:
    return SimpleNamespace(time=time, to_image=lambda: Image.new("RGB", (32, 32), (grey, grey, grey)))


def _fake_av(monkeypatch: pytest.MonkeyPatch, frames: list[Any], rate: int = 24) -> None:
    stream = SimpleNamespace(average_rate=Fraction(rate), time_base=Fraction(1, 1000))
    container = SimpleNamespace(streams=SimpleNamespace(video=[stream]), decode=lambda _stream: iter(frames))

    @contextmanager
    def open_container(_path: str) -> Iterator[Any]:
        yield container

    monkeypatch.setattr(media, "_require_av", lambda: SimpleNamespace(open=open_container))


def _greys(frames: Any) -> list[int]:
    return [int(value) for value in frames[:, 0, 0, 0]]


def test_frames_without_a_timestamp_are_placed_on_the_frame_grid_by_their_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_av(monkeypatch, [_frame(10, None), _frame(20, None), _frame(30, None)])

    assert _greys(read_video_frames(Path("c.avi"), 32, 100)) == [10, 20, 30]


def test_frames_are_timed_from_the_first_one_not_from_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    times = [5.0, 5.0 + 1 / 12, 5.0 + 2 / 12]  # every other 24 fps tick, starting at 5 s
    _fake_av(monkeypatch, [_frame(10, times[0]), _frame(20, times[1]), _frame(30, times[2])])

    assert _greys(read_video_frames(Path("c.avi"), 32, 100)) == [10, 10, 20, 20, 30]


@pytest.mark.parametrize(
    ("rate", "time_base", "expected_duration", "expected_snap"),
    [
        (Fraction(30), Fraction(1, 1000), 1 / 30, 0.001),
        (None, Fraction(1, 1000), 1 / 24, 0.001),  # no average rate: H3's 24 fps
        (Fraction(24), None, 1 / 24, 0.0),  # no time base: no snapping
        (Fraction(24), Fraction(1, 90), 1 / 24, 1 / 24 / 4),  # a coarse time base snaps at most a quarter frame
    ],
)
def test_the_frame_duration_and_the_largest_snap(
    rate: Fraction | None, time_base: Fraction | None, expected_duration: float, expected_snap: float
) -> None:
    stream = SimpleNamespace(average_rate=rate, time_base=time_base)

    assert _frame_timing(stream) == pytest.approx((expected_duration, expected_snap))


def test_a_time_exactly_one_snap_from_the_grid_still_moves_onto_it() -> None:
    assert _on_grid(0.25, 1.0, 0.25) == 0.0
