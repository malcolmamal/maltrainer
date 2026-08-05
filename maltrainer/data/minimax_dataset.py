"""
Dataset loader for MiniMax H3 image datasets (no caption files required).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def align_to_32(val: int) -> int:
    return ((val + 31) // 32) * 32


class MiniMaxH3ImageDataset(Dataset):
    """Image-only dataset loader for MiniMax H3 training."""

    def __init__(
        self,
        data_root: Union[str, Path],
        resolution: Union[int, Tuple[int, int]] = 512,
        trigger_token: Optional[str] = None,
        default_caption: str = "",
        repeats: int = 1,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.repeats = max(repeats, 1)
        self.trigger_token = (trigger_token or "").strip()
        self.default_caption = default_caption.strip()

        if isinstance(resolution, int):
            self.height = align_to_32(resolution)
            self.width = align_to_32(resolution)
        else:
            self.height = align_to_32(resolution[0])
            self.width = align_to_32(resolution[1])

        self.image_paths: List[Path] = []
        self.captions: List[str] = []
        self._load_dataset()

        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),  # Convert PIL Image to [3, H, W] in [0.0, 1.0]
        ])

    def _load_dataset(self):
        if not self.data_root.exists():
            logger.warning("Dataset root does not exist: %s", self.data_root)
            return

        base_images: List[Path] = []
        for root, _, files in os.walk(self.data_root):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    base_images.append(Path(root) / file)

        base_images.sort()

        for img_path in base_images:
            txt_path = img_path.with_suffix(".txt")
            caption = ""
            if txt_path.exists():
                try:
                    caption = txt_path.read_text(encoding="utf-8").strip()
                except Exception as e:
                    logger.warning("Error reading caption for %s: %s", img_path, e)

            if not caption:
                if self.trigger_token and self.default_caption and self.trigger_token not in self.default_caption:
                    caption = f"{self.trigger_token} {self.default_caption}"
                elif self.trigger_token:
                    caption = self.trigger_token
                elif self.default_caption:
                    caption = self.default_caption
                else:
                    caption = ""

            for _ in range(self.repeats):
                self.image_paths.append(img_path)
                self.captions.append(caption)

        logger.info("Loaded %d samples (%d unique images x %d repeats) from %s", len(self.image_paths), len(base_images), self.repeats, self.data_root)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        img_path = self.image_paths[index]
        caption = self.captions[index]

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(image)
        except Exception as e:
            logger.error("Failed to load image %s: %s", img_path, e)
            pixel_values = torch.zeros((3, self.height, self.width), dtype=torch.float32)

        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "image_path": str(img_path),
        }
