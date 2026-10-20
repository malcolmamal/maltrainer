"""Decoding a clip's frames and a file's soundtrack with PyAV, onto H3's 24 fps video clock and
32 kHz stereo audio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from maltrainer.data.minimax_dataset import AUDIO_SAMPLE_RATE, FPS, bucket_size, resize_and_crop


def _require_av() -> Any:
    try:
        import av
    except ImportError as e:
        raise ImportError("Video and audio training items need PyAV: pip install av") from e
    return av


def _on_grid(time: float, frame_duration: float, snap: float) -> float:
    """``time`` moved onto the clip's constant frame grid when it is within ``snap`` of it.

    Containers round timestamps to their time base: MKV / WebM keep whole milliseconds, so 1/24 s
    is stored as 42 ms, past the 24 fps tick it belongs to. A frame far from the grid (variable
    frame rate) keeps its own time."""
    grid = round(time / frame_duration) * frame_duration
    return grid if abs(time - grid) <= snap else time


class _TickClock:
    """Repeats the latest source frame onto every 24 fps tick, converting (resizing) each frame once."""

    def __init__(self, resolution: int, max_frames: int) -> None:
        self.resolution = resolution
        self.max_frames = max_frames
        self.frames: list[np.ndarray] = []
        self.size: tuple[int, int] | None = None
        self.last_frame: Any = None
        self.last_array: np.ndarray | None = None
        self.last_time = 0.0

    @property
    def full(self) -> bool:
        return len(self.frames) >= self.max_frames

    def _convert(self, frame: Any) -> np.ndarray:
        image = frame.to_image()
        if self.size is None:
            self.size = bucket_size(image.width, image.height, self.resolution)
        return np.asarray(resize_and_crop(image, self.size), dtype=np.uint8)

    def emit_until(self, end: float) -> None:
        """Every 24 fps tick strictly before ``end`` belongs to the latest frame."""
        while self.last_frame is not None and not self.full and len(self.frames) / FPS < end - 1e-6:
            if self.last_array is None:
                self.last_array = self._convert(self.last_frame)
            self.frames.append(self.last_array)

    def push(self, frame: Any, time: float) -> None:
        self.last_frame, self.last_array = frame, None
        self.last_time = time


def _frame_timing(stream: Any) -> tuple[float, float]:
    """The frame duration and the largest timestamp rounding to snap back onto the frame grid."""
    rate = stream.average_rate
    frame_duration = 1.0 / float(rate) if rate else 1.0 / FPS
    return frame_duration, min(float(stream.time_base or 0.0), frame_duration / 4)


def read_video_frames(path: Path, resolution: int, max_frames: int) -> np.ndarray:
    """Up to ``max_frames`` frames resampled to 24 fps, as uint8 [T, H, W, 3].

    Resampling is by timestamp (each 24 fps tick takes the latest source frame at or before
    it), so 30 / 60 fps footage keeps real-time speed instead of playing fast. Frames are
    resized as they are decoded, so a long 4K source never sits in memory at full size.
    """
    av = _require_av()
    clock = _TickClock(resolution, max_frames)
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path.name}: no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frame_duration, snap = _frame_timing(stream)

        origin: float | None = None
        for decoded, frame in enumerate(container.decode(stream)):
            time = frame.time if frame.time is not None else decoded * frame_duration
            if origin is None:
                origin = time
            time = _on_grid(time - origin, frame_duration, snap)
            clock.emit_until(time)
            if clock.full:
                break
            clock.push(frame, time)

        clock.emit_until(clock.last_time + frame_duration)

    if not clock.frames:
        raise ValueError(f"{path.name}: no decodable video frames")
    return np.stack(clock.frames)


def read_audio(path: Path) -> np.ndarray | None:
    """The first audio stream as float32 stereo [2, L] at 32 kHz in [-1, 1], or None.

    None means "no sound to train": no audio stream, or a track that decodes to digital
    silence. Mono is duplicated into both channels at its own level, and surround is
    downmixed by FFmpeg's resampler.
    """
    av = _require_av()
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        # Mono stays mono here and is duplicated below: FFmpeg's mono -> stereo upmix is -3 dB.
        layout = "mono" if stream.channels == 1 else "stereo"
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=AUDIO_SAMPLE_RATE)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            frame.pts = None  # let the resampler run on sample count, not container timestamps
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())

    if not chunks:
        return None
    wav = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    if not np.any(wav):
        return None
    return np.clip(wav[:2], -1.0, 1.0)
