"""Decoding a clip's frames and a file's soundtrack with PyAV, onto H3's 24 fps video clock and
32 kHz stereo audio."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from maltrainer.data.minimax_dataset import AUDIO_SAMPLE_RATE, FPS, bucket_size, resize_and_crop


def _require_av():
    try:
        import av  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("Video and audio training items need PyAV: pip install av") from e
    return av


def _on_grid(time: float, frame_duration: float, snap: float) -> float:
    """``time`` moved onto the clip's constant frame grid when it is within ``snap`` of it.

    Containers round timestamps to their time base: MKV / WebM keep whole milliseconds, so 1/24 s
    is stored as 42 ms, past the 24 fps tick it belongs to. A frame far from the grid (variable
    frame rate) keeps its own time."""
    grid = round(time / frame_duration) * frame_duration
    return grid if abs(time - grid) <= snap else time


def read_video_frames(path: Path, resolution: int, max_frames: int) -> np.ndarray:
    """Up to ``max_frames`` frames resampled to 24 fps, as uint8 [T, H, W, 3].

    Resampling is by timestamp (each 24 fps tick takes the latest source frame at or before
    it), so 30 / 60 fps footage keeps real-time speed instead of playing fast. Frames are
    resized as they are decoded, so a long 4K source never sits in memory at full size.
    """
    av = _require_av()
    frames: List[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path.name}: no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = stream.average_rate
        frame_duration = 1.0 / float(rate) if rate else 1.0 / FPS
        snap = min(float(stream.time_base or 0.0), frame_duration / 4)

        size: Optional[Tuple[int, int]] = None
        origin: Optional[float] = None
        last_frame = None
        last_array: Optional[np.ndarray] = None
        last_time = 0.0
        target = 0

        def convert(frame) -> np.ndarray:
            nonlocal size
            image = frame.to_image()
            if size is None:
                size = bucket_size(image.width, image.height, resolution)
            return np.asarray(resize_and_crop(image, size), dtype=np.uint8)

        for decoded, frame in enumerate(container.decode(stream)):
            time = frame.time if frame.time is not None else decoded * frame_duration
            if origin is None:
                origin = time
            time = _on_grid(time - origin, frame_duration, snap)
            # every 24 fps tick strictly before this frame belongs to the previous one
            while last_frame is not None and target < max_frames and target / FPS < time - 1e-6:
                if last_array is None:
                    last_array = convert(last_frame)
                frames.append(last_array)
                target += 1
            if target >= max_frames:
                break
            last_frame, last_array = frame, None
            last_time = time

        if last_frame is not None and target < max_frames:
            end = last_time + frame_duration
            while target < max_frames and target / FPS < end - 1e-6:
                if last_array is None:
                    last_array = convert(last_frame)
                frames.append(last_array)
                target += 1

    if not frames:
        raise ValueError(f"{path.name}: no decodable video frames")
    return np.stack(frames)


def read_audio(path: Path) -> Optional[np.ndarray]:
    """The first audio stream as float32 stereo [2, L] at 32 kHz in [-1, 1], or None.

    None means "no sound to train": no audio stream, or a track that decodes to digital
    silence. Mono is duplicated and surround is downmixed by FFmpeg's resampler.
    """
    av = _require_av()
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_SAMPLE_RATE)
        chunks: List[np.ndarray] = []
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
