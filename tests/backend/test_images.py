from __future__ import annotations

from pathlib import Path

from PIL import Image

from backend.app import count_images, scan_images


def _image(path: Path, size: tuple[int, int] = (20, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def _dataset(root: Path) -> Path:
    _image(root / "a.png")
    _image(root / "people" / "b.png", (40, 20))
    _image(root / ".ipynb_checkpoints" / "a-checkpoint.png", (40, 10))
    _image(root / "people" / ".trash" / "c.png")
    return root


def test_the_ui_counts_only_images_outside_hidden_folders(tmp_path: Path) -> None:
    assert count_images(_dataset(tmp_path / "dataset")) == 2


def test_the_ui_scans_only_images_outside_hidden_folders(tmp_path: Path) -> None:
    assert scan_images(_dataset(tmp_path / "dataset")) == (2, 1, (40, 20))


def test_a_missing_folder_has_no_images(tmp_path: Path) -> None:
    assert count_images(tmp_path / "absent") == 0
