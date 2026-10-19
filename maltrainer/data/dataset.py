"""
Dataset for Z-Image training.
"""

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from maltrainer.data.walk import walk_visible


class ZImageDataset(Dataset):
    """Dataset for Z-Image training with VAE encoding and text encoding."""

    def __init__(
        self,
        data_root: str,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = False,
        trigger_token: str | None = None,
        repeats: int = 1,
    ):
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.trigger_token = trigger_token
        self.repeats = repeats

        # Image transforms
        t_list = [transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)]
        if center_crop:
            t_list.append(transforms.CenterCrop(resolution))
        if random_flip:
            t_list.append(transforms.RandomHorizontalFlip())

        t_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transforms.Compose(t_list)

        # Find all images
        self.image_paths = []
        self.captions = []

        files = (Path(folder) / name for folder, _, names in walk_visible(self.data_root) for name in names)
        for img_path in files:
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                self.image_paths.append(img_path)

                # Look for caption file
                caption_path = img_path.with_suffix(".txt")
                if caption_path.exists():
                    with Path(caption_path).open("r", encoding="utf-8") as f:
                        caption = f.read().strip()
                else:
                    caption = ""

                # Add trigger token if specified
                if trigger_token:
                    caption = f"{trigger_token} {caption}" if caption else trigger_token

                self.captions.append(caption)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_root}")

        # Apply repeats
        if self.repeats > 1:
            self.image_paths = self.image_paths * self.repeats
            self.captions = self.captions * self.repeats

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.transform(image)

        return {
            "pixel_values": pixel_values,
            "caption": self.captions[idx],
        }
