"""Tiny media files for the H3 trainer: still images, FFV1 clips (optionally with sound) and WAV recordings."""

from __future__ import annotations

import zlib
from pathlib import Path

import av
import numpy as np
from PIL import Image


def write_image(path: Path, caption: str | None = None, size: tuple[int, int] = (32, 32)) -> Path:
    """A solid image whose colour follows its name, with a ``.txt`` caption when one is given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seed = zlib.crc32(path.name.encode("utf-8"))
    Image.new("RGB", size, (seed % 256, seed // 256 % 256, seed // 65536 % 256)).save(path)
    if caption is not None:
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return path


def sound(seconds: float, rate: int = 16000, seed: int = 0, channels: int = 2) -> np.ndarray:
    """Deterministic noise shaped [channels, samples]."""
    samples = round(seconds * rate)
    return np.random.default_rng(seed).uniform(-0.5, 0.5, (channels, samples)).astype(np.float32)


def _add_sound(container: av.container.OutputContainer, samples: np.ndarray, rate: int) -> av.AudioStream:
    stream = container.add_stream("pcm_f32le", rate=rate)
    stream.layout = "mono" if samples.shape[0] == 1 else "stereo"
    return stream


def _encode_sound(container: av.container.OutputContainer, stream: av.AudioStream, samples: np.ndarray) -> None:
    interleaved = np.ascontiguousarray(samples.T.reshape(1, -1), dtype=np.float32)
    frame = av.AudioFrame.from_ndarray(interleaved, format="flt", layout=stream.layout.name)
    frame.sample_rate = stream.rate
    for packet in [*stream.encode(frame), *stream.encode()]:
        container.mux(packet)


def write_wav(path: Path, samples: np.ndarray, rate: int = 16000, caption: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = _add_sound(container, samples, rate)
        container.start_encoding()
        _encode_sound(container, stream, samples)
    if caption is not None:
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return path


def write_clip(
    path: Path,
    frames: int = 5,
    audio: np.ndarray | None = None,
    caption: str | None = None,
    size: tuple[int, int] = (32, 32),
    rate: int = 16000,
) -> Path:
    """A lossless 24 fps AVI of ``frames`` grey frames, with the ``audio`` soundtrack when given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        video = container.add_stream("ffv1", rate=24)
        video.width, video.height = size
        video.pix_fmt = "bgr0"
        track = _add_sound(container, audio, rate) if audio is not None else None
        container.start_encoding()
        for index in range(frames):
            pixels = np.full((size[1], size[0], 3), index * 10 % 256, dtype=np.uint8)
            for packet in video.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if track is not None and audio is not None:
            _encode_sound(container, track, audio)
    if caption is not None:
        path.with_suffix(".txt").write_text(caption, encoding="utf-8")
    return path
