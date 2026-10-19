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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

from maltrainer.data.walk import walk_visible

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
# A caption prefix for every image below it, e.g. "photo of [folder]" (docs/datasets.md).
PREFIX_FILE = "prefix.txt"


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
    return "audio" if ext in SUPPORTED_AUDIO_EXTENSIONS else None


def _read_text(path: Path, one_line: bool = False) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
        return " ".join(text.split()) if one_line else text.strip()
    except Exception as e:
        logger.warning("Error reading %s: %s", path, e)
        return ""


def _prefix_for(folder: Path, data_root: Path, templates: Dict[Path, Optional[str]]) -> str:
    """The nearest ``prefix.txt`` from ``folder`` up to one level above the dataset folder,
    with ``[folder]`` replaced by ``folder``'s name; "" if there is none. An empty prefix.txt
    counts: it switches off any prefix from further up.

    ``templates`` memoizes each directory's prefix.txt text (None = no file) across a scan.
    """
    stop = data_root.parent
    current = folder
    while True:
        if current not in templates:
            path = current / PREFIX_FILE
            templates[current] = _read_text(path, one_line=True) if path.exists() else None
        if templates[current] is not None:
            return templates[current].replace("[folder]", folder.name)
        if current == stop or current.parent == current:
            return ""
        current = current.parent


def _join_caption(prefix: str, caption: str) -> str:
    prefix = prefix.strip().rstrip(",").strip()
    if not prefix:
        return caption
    if not caption:
        return prefix
    return caption if caption.startswith(prefix) else f"{prefix}, {caption}"


def _caption_for(path: Path, trigger_token: str, default_caption: str, prefix: str = "") -> str:
    """The sidecar ``<stem>.txt`` behind ``prefix``; without one, the prefix alone (it names the
    subject); without either, the trigger / default caption."""
    caption = _read_text(path.with_suffix(".txt"))
    if caption or prefix:
        return _join_caption(prefix, caption)

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
    templates: Dict[Path, Optional[str]] = {}
    for root, _, files in walk_visible(data_root):
        for name in files:
            path = Path(root) / name
            kind = _media_kind(path)
            if kind in kinds:
                prefix = _prefix_for(path.parent, data_root, templates)
                sources.append(MediaSource(path, kind, _caption_for(path, trigger_token, default_caption, prefix)))
    sources.sort(key=lambda source: str(source.path))
    return sources


def count_media_files(data_root: Union[str, Path], kinds: Iterable[str] = MEDIA_KINDS) -> int:
    """Number of supported files, for the dry run's step estimate (no decoding)."""
    data_root = Path(data_root)
    if not data_root.exists():
        return 0
    kinds = set(kinds)
    found = (_media_kind(Path(root) / name) for root, _, files in walk_visible(data_root) for name in files)
    return sum(1 for kind in found if kind in kinds)


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


def plan_segments(total_frames: int, max_frames: int, mode: str, min_frames: int) -> List[Tuple[int, int]]:
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
    window = wav[:, start : start + want]
    if window.shape[1] < want:
        window = np.pad(window, ((0, 0), (0, want - window.shape[1])))
    return window


def frames_tensor(frames: np.ndarray) -> torch.Tensor:
    """uint8 [T, H, W, 3] -> float [1, 3, T, H, W] in [0, 1]."""
    return torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float().div_(255.0)


def summarize_kinds(sources: Sequence[MediaSource]) -> str:
    counts = {kind: sum(1 for source in sources if source.kind == kind) for kind in MEDIA_KINDS}
    return ", ".join(f"{count} {kind}{'s' if count != 1 else ''}" for kind, count in counts.items())
