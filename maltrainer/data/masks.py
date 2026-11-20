"""Masked training's files: a grayscale picture beside a training image that says which part counts.

A mask is named after its image with OneTrainer's suffix (``photo.jpg`` -> ``photo-masklabel.png``, any image
format), so masks made by other trainers work as they are. White trains fully, black at the floor the config sets.
The mask is read at train time onto the latent grid with the crop the picture got when it was cached.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from maltrainer.data.minimax_dataset import resize_and_crop

MASK_SUFFIX = "-masklabel"
MASK_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ASPECT_TOLERANCE = 0.02  # a mask whose shape is further off than this from its picture's is not used
MIN_COVER = 0.01  # a mask with less white than this leaves nothing to train on


def is_mask_file(path: str | Path) -> bool:
    return Path(path).stem.lower().endswith(MASK_SUFFIX)


def mask_for(image_path: str | Path) -> Path | None:
    """The mask file beside an image (any extension, any case), or None."""
    image = Path(image_path)
    wanted = {(image.stem + MASK_SUFFIX + ext).lower() for ext in MASK_EXTENSIONS}
    for sibling in sorted(image.parent.iterdir()):
        if sibling.name.lower() in wanted:
            return sibling
    return None


def read_mask(path: str | Path) -> Image.Image:
    """A mask file as grayscale (L): white = train. Transparent counts as black (a white subject painted on an empty
    layer); a 16-bit file keeps its range."""
    with Image.open(path) as image:
        image.load()
        if image.mode in ("I;16", "I;16B", "I;16L", "I"):
            data = np.asarray(image, dtype=np.float64)
            return Image.fromarray(np.clip(data / (257.0 if np.max(data) > 255 else 1.0), 0, 255).astype(np.uint8), "L")
        if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            flat: Image.Image = Image.alpha_composite(Image.new("RGBA", rgba.size, (0, 0, 0, 255)), rgba)
            return flat.convert("L")
        gray: Image.Image = image.convert("L")
        return gray


def aspect_ok(mask_size: tuple[int, int], image_size: tuple[int, int]) -> bool:
    """Is a mask (w, h) its picture's shape? A smaller one is fine (stretched); a cropped or rotated one is not."""
    (mw, mh), (iw, ih) = mask_size, image_size
    return min(mw, mh, iw, ih) > 0 and abs((mw / mh) / (iw / ih) - 1.0) <= ASPECT_TOLERANCE


def mask_problem(path: str | Path, image_size: tuple[int, int]) -> str:
    """Why a mask can't be used for a picture of ``image_size`` (w, h), or "" when it can."""
    try:
        mask = read_mask(path)
    except OSError:
        return "can't be read"
    if not aspect_ok(mask.size, image_size):
        return "isn't the shape of its picture (cropped or rotated after the mask was made?)"
    if float(np.asarray(mask, dtype=np.float32).mean()) / 255.0 < MIN_COVER:
        return "is (almost) all black: it leaves nothing to train on"
    return ""


def latent_mask(
    path: str | Path, image_size: tuple[int, int], bucket: tuple[int, int], grid: tuple[int, int]
) -> torch.Tensor:
    """The mask on a latent grid: [1, 1, h, w] in [0, 1]. ``image_size`` (w, h) is the picture as stored, ``bucket``
    (w, h) the size it was cropped to for the VAE, ``grid`` (h, w) its latent."""
    mask = read_mask(path)
    if mask.size != image_size:
        mask = mask.resize(image_size, Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.asarray(resize_and_crop(mask, bucket), dtype=np.float32).copy()) / 255.0
    return torch.nn.functional.adaptive_avg_pool2d(pixels[None, None], grid).clamp_(0.0, 1.0)
