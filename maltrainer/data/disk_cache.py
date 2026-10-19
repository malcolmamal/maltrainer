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
from pathlib import Path
from typing import Optional, Union

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".maltrainer_cache"


def content_hash(path: Union[str, Path]) -> str:
    """BLAKE2b of the file's bytes: the same image matches wherever it is."""
    digest = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_identity(path: Union[str, Path, None]) -> str:
    """File name and size, so the same model file matches from any folder or machine."""
    if not path:
        return "none"
    path = Path(path)
    return f"{path.name}|{path.stat().st_size}" if path.exists() else path.name


def cache_key(*parts: object) -> str:
    return hashlib.sha1("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()


class FolderCache:
    """The ``.maltrainer_cache`` of one image folder."""

    def __init__(self, folder: Union[str, Path]):
        self.root = Path(folder) / CACHE_DIR_NAME

    def _path(self, kind: str, key: str) -> Path:
        return self.root / kind / f"{key}.safetensors"

    def has(self, kind: str, key: str) -> bool:
        return self._path(kind, key).exists()

    def get(self, kind: str, key: str) -> Optional[torch.Tensor]:
        path = self._path(kind, key)
        if not path.exists():
            return None
        try:
            return load_file(str(path))["tensor"]
        except Exception as e:  # a truncated or foreign file: re-encode it
            logger.warning("Unreadable cache entry %s (%s); re-encoding", path, e)
            return None

    def put(self, kind: str, key: str, tensor: torch.Tensor) -> None:
        path = self._path(kind, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        save_file({"tensor": tensor.detach().cpu().contiguous()}, str(tmp))
        tmp.replace(path)
