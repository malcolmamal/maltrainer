"""
Media loading for MiniMax H3 training: stills, video clips (with their soundtrack) and
audio-only recordings, all from one folder.

H3 is a joint audio+video model, and its inference layout (ComfyUI's PackedLayout) always
carries both streams. The trainer therefore turns every source into the same shape of
sample -- a video latent plus audio rows -- and only the loss changes with the kind:

    image   1 latent frame; audio rows are noised silence with no audio loss
    clip    17k + 5 frames at 24 fps; audio rows are its soundtrack (or silence when the
            clip has no audio track, is digitally silent, or its stem ends in ``_mute``)
    audio   a zeros video placeholder sized to the recording; the video loss is dropped
            and the audio is the training target (Fizgig's "voice" items)

Captions come from a sidecar ``<stem>.txt`` next to each file, with the same trigger /
default-caption fallback the image trainer always used.

Decoding uses PyAV (``pip install av``), which bundles FFmpeg, so any common container,
sample rate or channel layout works. It is imported lazily: an images-only dataset never
needs it.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
MEDIA_KINDS = ("image", "clip", "audio")

FPS = 24  # H3's video clock
AUDIO_SAMPLE_RATE = 32000
AUDIO_HOP = 800  # audio VAE samples per latent frame -> 40 latent frames per second
AUDIO_LATENTS_PER_SECOND = AUDIO_SAMPLE_RATE // AUDIO_HOP

# Shortest usable lengths, in 24 fps frames. A 5-frame audio item is 0.2 s of sound, which
# trains nothing worth a step (Fizgig excludes it for the same reason).
MIN_CLIP_FRAMES = 5
MIN_AUDIO_FRAMES = 22

MUTE_SUFFIX = "_mute"


def align_to_32(val: int) -> int:
    return ((val + 31) // 32) * 32


def grid_frames(frames: int) -> int:
    """Largest H3 clip length <= frames on the 17k + 5 grid (5, 22, 39, ...), or 0."""
    if frames < 5:
        return 0
    return 5 + ((frames - 5) // 17) * 17


def audio_latent_frames(frames: int) -> int:
    """Audio latent frames that accompany ``frames`` video frames (ComfyUI's temporal_shape).

    Rounded from the duration, so a still (1 frame) still carries 2 audio latents, and
    56 / 107 frame clips get round(), not the ceil() the audio VAE would give their samples.
    """
    return round(frames / FPS * AUDIO_LATENTS_PER_SECOND)


def bucket_size(width: int, height: int, resolution: int, step: int = 32) -> Tuple[int, int]:
    """Aspect-preserving target size whose area is about ``resolution``^2, on a ``step`` grid.

    32 px = the VAE's 16x downsample times the DiT's 2x2 patch, so every side patchifies exactly.
    """
    scale = math.sqrt((resolution * resolution) / float(width * height))
    target_w = max(step, int(width * scale) // step * step)
    target_h = max(step, int(height * scale) // step * step)
    return target_w, target_h


def resize_and_crop(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Scale to cover ``size`` (Lanczos), then centre-crop to it exactly."""
    target_w, target_h = size
    scale = max(target_w / image.width, target_h / image.height)
    resized_w = max(target_w, round(image.width * scale))
    resized_h = max(target_h, round(image.height * scale))
    if (resized_w, resized_h) != image.size:
        image = image.resize((resized_w, resized_h), Image.LANCZOS)
    left = (resized_w - target_w) // 2
    top = (resized_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


@dataclass
class MediaSource:
    """One file in the dataset folder."""

    path: Path
    kind: str  # "image" | "clip" | "audio"
    caption: str

    @property
    def muted(self) -> bool:
        return self.path.stem.lower().endswith(MUTE_SUFFIX)


def _media_kind(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if ext in SUPPORTED_VIDEO_EXTENSIONS:
        return "clip"
    if ext in SUPPORTED_AUDIO_EXTENSIONS:
        return "audio"
    return None


def _caption_for(path: Path, trigger_token: str, default_caption: str) -> str:
    txt_path = path.with_suffix(".txt")
    if txt_path.exists():
        try:
            caption = txt_path.read_text(encoding="utf-8").strip()
            if caption:
                return caption
        except Exception as e:
            logger.warning("Error reading caption for %s: %s", path, e)

    if trigger_token and default_caption:
        # "a photo of sks" already names the subject; otherwise prepend the trigger to it
        if trigger_token in default_caption:
            return default_caption
        return f"{trigger_token} {default_caption}"
    return trigger_token or default_caption or ""


def scan_media_sources(
    data_root: Union[str, Path],
    kinds: Iterable[str] = MEDIA_KINDS,
    trigger_token: Optional[str] = None,
    default_caption: str = "",
) -> List[MediaSource]:
    """Every supported file under ``data_root`` whose kind is in ``kinds``, sorted by path."""
    data_root = Path(data_root)
    kinds = set(kinds)
    unknown = kinds - set(MEDIA_KINDS)
    if unknown:
        raise ValueError(f"Unknown dataset media kind(s) {sorted(unknown)}; choose from {MEDIA_KINDS}")
    if not data_root.exists():
        logger.warning("Dataset root does not exist: %s", data_root)
        return []

    trigger_token = (trigger_token or "").strip()
    default_caption = (default_caption or "").strip()
    sources: List[MediaSource] = []
    for root, _, files in os.walk(data_root):
        for name in files:
            path = Path(root) / name
            kind = _media_kind(path)
            if kind in kinds:
                sources.append(MediaSource(path, kind, _caption_for(path, trigger_token, default_caption)))
    sources.sort(key=lambda source: str(source.path))
    return sources


def count_media_files(data_root: Union[str, Path], kinds: Iterable[str] = MEDIA_KINDS) -> int:
    """Number of supported files, for the dry run's step estimate (no decoding)."""
    data_root = Path(data_root)
    if not data_root.exists():
        return 0
    kinds = set(kinds)
    return sum(
        1
        for root, _, files in os.walk(data_root)
        for name in files
        if _media_kind(Path(root) / name) in kinds
    )


# --- decoding -----------------------------------------------------------------------


def load_image(path: Path, resolution: int, aspect_mode: str = "bucket") -> torch.Tensor:
    """RGB [3, H, W] in [0, 1].

    ``bucket`` keeps the aspect ratio (area ~ resolution^2, sides on the 32 px grid, centre
    crop). ``stretch`` is the trainer's original behaviour: a non-uniform resize to a
    resolution x resolution square, which distorts any non-square photo.
    """
    image = Image.open(path).convert("RGB")
    if aspect_mode == "stretch":
        side = align_to_32(resolution)
        image = image.resize((side, side), Image.BICUBIC)
    else:
        image = resize_and_crop(image, bucket_size(image.width, image.height, resolution))
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).float().div_(255.0)


def _require_av():
    try:
        import av  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Video and audio training items need PyAV: pip install av"
        ) from e
    return av


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

        size: Optional[Tuple[int, int]] = None
        origin: Optional[float] = None
        last_frame = None
        last_array: Optional[np.ndarray] = None
        last_time = 0.0
        target = 0
        decoded = 0

        def convert(frame) -> np.ndarray:
            nonlocal size
            image = frame.to_image()
            if size is None:
                size = bucket_size(image.width, image.height, resolution)
            return np.asarray(resize_and_crop(image, size), dtype=np.uint8)

        for frame in container.decode(stream):
            time = frame.time if frame.time is not None else decoded * frame_duration
            decoded += 1
            if origin is None:
                origin = time
            time -= origin
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


def plan_segments(
    total_frames: int, max_frames: int, mode: str, min_frames: int
) -> List[Tuple[int, int]]:
    """(start_frame, frames) windows on the 17k + 5 grid.

    ``first`` keeps one window from the start (``min_frames`` or longer). ``split`` tiles the
    source with consecutive ``max_frames`` windows plus at most one shorter tail window, kept
    only if it reaches ``MIN_AUDIO_FRAMES`` -- otherwise a 21-frame remainder would shed a run
    of 5-frame (0.2 s) slivers.
    """
    max_frames = grid_frames(max_frames)
    if max_frames < min_frames:
        raise ValueError(f"max_frames must allow at least {min_frames} frames on the 17k + 5 grid")
    segments: List[Tuple[int, int]] = []
    start = 0
    while True:
        frames = grid_frames(min(max_frames, total_frames - start))
        floor = min_frames if not segments else max(min_frames, MIN_AUDIO_FRAMES)
        if frames < floor:
            break
        segments.append((start, frames))
        if mode != "split" or frames < max_frames:
            break
        start += frames
    return segments


def slice_audio(wav: np.ndarray, start_frame: int, frames: int) -> np.ndarray:
    """The hop-exact audio for a [start_frame, start_frame + frames) window.

    Exactly ``audio_latent_frames(frames) * 800`` samples (zero-padded at the end if the
    recording is short), so the VAE emits precisely the latent count the packed layout
    expects -- no per-channel trimming of encoded rows afterwards.
    """
    start = round(start_frame / FPS * AUDIO_SAMPLE_RATE)
    want = audio_latent_frames(frames) * AUDIO_HOP
    window = wav[:, start:start + want]
    if window.shape[1] < want:
        window = np.pad(window, ((0, 0), (0, want - window.shape[1])))
    return window


def frames_tensor(frames: np.ndarray) -> torch.Tensor:
    """uint8 [T, H, W, 3] -> float [1, 3, T, H, W] in [0, 1]."""
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().div_(255.0)


def summarize_kinds(sources: Sequence[MediaSource]) -> str:
    counts = {kind: sum(1 for source in sources if source.kind == kind) for kind in MEDIA_KINDS}
    return ", ".join(f"{count} {kind}{'s' if count != 1 else ''}" for kind, count in counts.items())
