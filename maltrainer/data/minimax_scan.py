"""Finding a dataset folder's media files and their captions (sidecar ``.txt`` plus ``prefix.txt``)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from maltrainer.data.walk import walk_visible

# The name this code logged under before it moved out of maltrainer/data/minimax_dataset.py.
logger = logging.getLogger("maltrainer.data.minimax_dataset")

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
MEDIA_KINDS = ("image", "clip", "audio")

MUTE_SUFFIX = "_mute"
# A caption prefix for every image below it, e.g. "photo of [folder]" (docs/datasets.md).
PREFIX_FILE = "prefix.txt"


@dataclass
class MediaSource:
    """One file in the dataset folder."""

    path: Path
    kind: str  # "image" | "clip" | "audio"
    caption: str

    @property
    def muted(self) -> bool:
        return self.path.stem.lower().endswith(MUTE_SUFFIX)


def _media_kind(path: Path) -> str | None:
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
    except (OSError, ValueError) as e:
        logger.warning("Error reading %s: %s", path, e)
        return ""


def _prefix_for(folder: Path, data_root: Path, templates: dict[Path, str | None]) -> str:
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
        if (template := templates[current]) is not None:
            return template.replace("[folder]", folder.name)
        if current in (stop, current.parent):
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
    data_root: str | Path,
    kinds: Iterable[str] = MEDIA_KINDS,
    trigger_token: str | None = None,
    default_caption: str = "",
) -> list[MediaSource]:
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
    sources: list[MediaSource] = []
    templates: dict[Path, str | None] = {}
    for root, _, files in walk_visible(data_root):
        for name in files:
            path = Path(root) / name
            kind = _media_kind(path)
            if kind in kinds:
                prefix = _prefix_for(path.parent, data_root, templates)
                sources.append(MediaSource(path, kind, _caption_for(path, trigger_token, default_caption, prefix)))
    sources.sort(key=lambda source: str(source.path))
    return sources


def count_media_files(data_root: str | Path, kinds: Iterable[str] = MEDIA_KINDS) -> int:
    """Number of supported files, for the dry run's step estimate (no decoding)."""
    data_root = Path(data_root)
    if not data_root.exists():
        return 0
    kinds = set(kinds)
    found = (_media_kind(Path(root) / name) for root, _, files in walk_visible(data_root) for name in files)
    return sum(1 for kind in found if kind in kinds)
