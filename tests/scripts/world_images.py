"""Small image folders for script tests."""

from __future__ import annotations

import zlib
from pathlib import Path

from PIL import Image


def write_dataset(folder: Path, count: int, size: tuple[int, int] = (64, 64)) -> Path:
    """``count`` solid images with a ``.txt`` caption each; their colour follows their name."""
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        seed = zlib.crc32(f"img{i}".encode())
        Image.new("RGB", size, (seed % 256, seed // 256 % 256, seed // 65536 % 256)).save(folder / f"img{i}.png")
        (folder / f"img{i}.txt").write_text(f"a photo of number {i}", encoding="utf-8")
    return folder
