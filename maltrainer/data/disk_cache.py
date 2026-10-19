"""
On-disk cache of encoded training inputs (VAE latents, caption embeddings), kept next to the
images: each folder that holds images gets a ``.maltrainer_cache/`` with the entries for its own
files, as ``.maltrainer_cache/<kind>/<key>.safetensors``.

Encoding is the slow part of a large dataset's setup: ~25 images/s through the VAE and ~3
captions/s through a 9B text encoder, repeated at the start of every run. Each entry is stored
once, under a key that hashes everything that determines it: the image's bytes (not its path),
the caption, the resolution, the encoder file. So a folder can be copied, moved or renamed and
keep its cache, editing an image or caption simply misses the old entry, and deleting a folder
deletes its cache with it. One file per tensor: a growing dataset only encodes its new images,
and an interrupted encode keeps what it finished.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".maltrainer_cache"


def content_hash(path: str | Path) -> str:
    """BLAKE2b of the file's bytes: the same image matches wherever it is."""
    digest = hashlib.blake2b(digest_size=20)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


IDENTITY_SAMPLE = 1 << 20


def model_identity(path: str | Path | None) -> str:
    """File name, size, and a hash of the first and last MiB: the same model file matches from
    any folder or machine, and two files of one size (two finetunes of a model) do not."""
    if not path:
        return "none"
    path = Path(path)
    if not path.exists():
        return path.name
    size = path.stat().st_size
    digest = hashlib.blake2b(digest_size=8)
    with Path(path).open("rb") as handle:
        head = handle.read(IDENTITY_SAMPLE)
        handle.seek(max(size - IDENTITY_SAMPLE, len(head)))
        digest.update(head + handle.read())
    return f"{path.name}|{size}|{digest.hexdigest()}"


def cache_key(*parts: object) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


class FolderCache:
    """The ``.maltrainer_cache`` of one image folder."""

    def __init__(self, folder: str | Path):
        self.root = Path(folder) / CACHE_DIR_NAME

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.safetensors"

    def has(self, kind: str, key: str) -> bool:
        return self._path(kind, key).exists()

    def get(self, kind: str, key: str) -> torch.Tensor | None:
        path = self._path(kind, key)
        if not path.exists():
            return None
        try:
            return load_file(str(path))["tensor"]
        except (OSError, ValueError, RuntimeError, KeyError, SafetensorError) as e:  # truncated or foreign: re-encode
            logger.warning("Unreadable cache entry %s (%s); re-encoding", path, e)
            return None

    def put(self, kind: str, key: str, tensor: torch.Tensor) -> None:
        path = self._path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A temp file of its own: two processes encoding the same entry must not write one file.
        handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{key}.", suffix=".tmp")
        os.close(handle)
        try:
            save_file({"tensor": tensor.detach().cpu().contiguous()}, tmp)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
