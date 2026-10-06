"""Image dataset for Supra2-IMG. Every sample is a 256x256 square."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPRA_IMAGE_SIZE = 256


def square_and_resize(image: Image.Image, size: int = SUPRA_IMAGE_SIZE) -> Image.Image:
    """Center-crop to a square, then resize. Larger and non-square images both become ``size``."""
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image


class Supra2ImgDataset(Dataset):
    def __init__(
        self,
        data_root: Union[str, Path],
        resolution: int = SUPRA_IMAGE_SIZE,
        trigger_token: Optional[str] = None,
        default_caption: str = "",
        repeats: int = 1,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.repeats = max(int(repeats), 1)
        self.trigger_token = (trigger_token or "").strip()
        self.default_caption = default_caption.strip()
        size = int(resolution)
        if size % 16 != 0:
            raise ValueError(f"Supra2-IMG resolution must be a multiple of 16, got {size}")
        if size != SUPRA_IMAGE_SIZE:
            logger.info(
                "Training at %dx%d. The checkpoint positions are for %d; they are interpolated.",
                size,
                size,
                SUPRA_IMAGE_SIZE,
            )
        self.size = size
        self.image_paths: List[Path] = []
        self.captions: List[str] = []
        self._load_dataset()
        self.to_tensor = transforms.ToTensor()

    def _load_dataset(self) -> None:
        if not self.data_root.exists():
            logger.warning("Dataset root does not exist: %s", self.data_root)
            return

        base_images: List[Path] = []
        for root, _, files in os.walk(self.data_root):
            for file in files:
                if Path(file).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    base_images.append(Path(root) / file)
        base_images.sort()

        for img_path in base_images:
            caption = self._caption_for(img_path)
            for _ in range(self.repeats):
                self.image_paths.append(img_path)
                self.captions.append(caption)

        logger.info(
            "Loaded %d samples (%d unique images x %d repeats) from %s; each image is resized to %dx%d",
            len(self.image_paths),
            len(base_images),
            self.repeats,
            self.data_root,
            self.size,
            self.size,
        )

    def _caption_for(self, img_path: Path) -> str:
        txt_path = img_path.with_suffix(".txt")
        caption = ""
        if txt_path.exists():
            try:
                caption = txt_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.warning("Error reading caption for %s: %s", img_path, exc)
        if caption:
            if self.trigger_token and self.trigger_token.lower() not in caption.lower():
                caption = f"{self.trigger_token} {caption}"
            return caption
        if self.trigger_token and self.default_caption and self.trigger_token.lower() not in self.default_caption.lower():
            return f"{self.trigger_token} {self.default_caption}"
        return self.trigger_token or self.default_caption

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_path = self.image_paths[index]
        caption = self.captions[index]
        try:
            image = square_and_resize(Image.open(img_path).convert("RGB"), self.size)
            pixel_values = self.to_tensor(image)
        except OSError as exc:
            logger.error("Failed to load image %s: %s", img_path, exc)
            pixel_values = torch.zeros((3, self.size, self.size), dtype=torch.float32)
        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "image_path": str(img_path),
        }
