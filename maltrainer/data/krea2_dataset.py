"""
Image loading for Krea 2: aspect buckets on a 16 px grid (VAE 8x times the DiT's 2x2 patch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from maltrainer.data.minimax_dataset import bucket_size, resize_and_crop

KREA2_GRID = 16


def load_krea2_image(path: Path, resolution: int, aspect_mode: str = "bucket") -> torch.Tensor:
    """RGB [3, H, W] in [0, 1]; ``bucket`` keeps aspect (area ~ resolution^2), ``square`` centre-crops."""
    image = Image.open(path).convert("RGB")
    if aspect_mode == "square":
        side = max(KREA2_GRID, resolution // KREA2_GRID * KREA2_GRID)
        size = (side, side)
    else:
        size = bucket_size(image.width, image.height, resolution, step=KREA2_GRID)
    image = resize_and_crop(image, size)
    return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
