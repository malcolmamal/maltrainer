"""
Image loading for FLUX.3: aspect buckets on a 16 px grid (the Flux2 VAE's 8x times its 2x2 latent fold).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from maltrainer.data.minimax_dataset import bucket_size, resize_and_crop

FLUX3_GRID = 16


def load_flux3_image(path: Path, resolution: int, aspect_mode: str = "bucket") -> torch.Tensor:
    """RGB [3, H, W] in [0, 1]. ``bucket`` keeps the aspect (area ~ resolution^2); ``square`` centre-crops."""
    image = Image.open(path).convert("RGB")
    if aspect_mode == "square":
        side = max(FLUX3_GRID, resolution // FLUX3_GRID * FLUX3_GRID)
        size = (side, side)
    else:
        size = bucket_size(image.width, image.height, resolution, step=FLUX3_GRID)
    image = resize_and_crop(image, size)
    return torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
