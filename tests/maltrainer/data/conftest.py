"""Tiny media files for the data loaders, written into a per-test ``data`` folder.

Images are solid colours (or two halves) so resize and crop results can be checked by hand.
Clips are lossless FFV1 in AVI, whose time base is one frame, and frame ``i`` is solid grey
``i * 20``, so a test can tell which source frame landed on which 24 fps tick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import av
import numpy as np
import pytest
from PIL import Image

RED = (255, 0, 0)
BLUE = (0, 0, 255)


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    return root


def _write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def write_file(data_root: Path) -> Callable[[str, str | bytes], Path]:
    """``write_file("people/ben.txt", "a caption")``; bytes are written as they are."""

    def write(rel: str, content: str | bytes) -> Path:
        return _write(data_root / rel, content)

    return write


@pytest.fixture
def make_image(data_root: Path) -> Callable[..., Path]:
    """A solid ``color`` image (left half ``color``, right half ``right`` if given), plus its caption."""

    def make(
        rel: str,
        size: Tuple[int, int] = (40, 24),
        color: Tuple[int, int, int] = RED,
        right: Optional[Tuple[int, int, int]] = None,
        caption: Optional[str] = None,
    ) -> Path:
        path = data_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", size, color)
        if right is not None:
            image.paste(right, (size[0] // 2, 0, size[0], size[1]))
        image.save(path)
        if caption is not None:
            _write(path.with_suffix(".txt"), caption)
        return path

    return make


def _add_audio(container: av.container.OutputContainer, samples: np.ndarray, rate: int) -> av.AudioStream:
    stream = container.add_stream("pcm_f32le", rate=rate)
    stream.layout = "mono" if samples.shape[0] == 1 else "stereo"
    return stream


def _encode_audio(container: av.container.OutputContainer, stream: av.AudioStream, samples: np.ndarray) -> None:
    if samples.shape[1] == 0:
        return
    interleaved = np.ascontiguousarray(samples.T.reshape(1, -1), dtype=np.float32)
    frame = av.AudioFrame.from_ndarray(interleaved, format="flt", layout=stream.layout.name)
    frame.sample_rate = stream.rate
    for packet in [*stream.encode(frame), *stream.encode()]:
        container.mux(packet)


@pytest.fixture
def make_wav(data_root: Path) -> Callable[..., Path]:
    """A float WAV of ``samples`` shaped [channels, length]; zero length writes an empty stream."""

    def make(rel: str, samples: np.ndarray, rate: int = 16000) -> Path:
        path = data_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), "w") as container:
            stream = _add_audio(container, samples, rate)
            container.start_encoding()
            _encode_audio(container, stream, samples)
        return path

    return make


@pytest.fixture
def make_clip(data_root: Path) -> Callable[..., Path]:
    """An FFV1 AVI of ``frames`` solid grey frames at ``fps``, with an optional soundtrack."""

    def make(
        rel: str,
        frames: int = 6,
        fps: int = 24,
        size: Tuple[int, int] = (32, 32),
        audio: Optional[np.ndarray] = None,
        rate: int = 16000,
    ) -> Path:
        path = data_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), "w") as container:
            video = container.add_stream("ffv1", rate=fps)
            video.width, video.height = size
            video.pix_fmt = "bgr0"
            sound = _add_audio(container, audio, rate) if audio is not None else None
            container.start_encoding()
            for index in range(frames):
                pixels = np.full((size[1], size[0], 3), index * 20, dtype=np.uint8)
                for packet in video.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            if sound is not None:
                _encode_audio(container, sound, audio)
        return path

    return make
