from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from maltrainer.data.minimax_dataset import scan_media_sources

Write = Callable[[str, "str | bytes"], Path]


def _captions(data_root: Path, **kwargs: str) -> dict[str, str]:
    return {s.path.relative_to(data_root).as_posix(): s.caption for s in scan_media_sources(data_root, **kwargs)}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_a_prefix_applies_below_its_folder_with_each_images_own_folder_name(data_root: Path, write_file: Write) -> None:
    write_file("people/prefix.txt", "photo of [folder]")
    write_file("people/ben/a.png", "")
    write_file("people/ben/a.txt", "smiling")
    write_file("people/ann/b.png", "")
    write_file("c.png", "")
    write_file("c.txt", "a tree")

    captions = _captions(data_root)

    assert captions == {
        "c.png": "a tree",
        "people/ann/b.png": "photo of ann",
        "people/ben/a.png": "photo of ben, smiling",
    }


def test_a_prefix_without_a_sidecar_names_the_subject_instead_of_the_trigger(
    data_root: Path, write_file: Write
) -> None:
    write_file("prefix.txt", "photo of [folder]")
    write_file("a.png", "")

    assert _captions(data_root, trigger_token="sks", default_caption="a photo") == {"a.png": "photo of data"}


def test_the_nearest_prefix_wins_and_an_empty_one_switches_prefixes_off(data_root: Path, write_file: Write) -> None:
    write_file("prefix.txt", "outer")
    write_file("near/prefix.txt", "inner")
    write_file("near/a.png", "")
    write_file("near/a.txt", "cat")
    write_file("off/prefix.txt", "  \n")
    write_file("off/b.png", "")
    write_file("off/b.txt", "dog")
    write_file("c.png", "")
    write_file("c.txt", "owl")

    assert _captions(data_root) == {"c.png": "outer, owl", "near/a.png": "inner, cat", "off/b.png": "dog"}


@pytest.mark.parametrize(
    ("prefix", "caption", "expected"),
    [
        ("photo of x,", "smiling", "photo of x, smiling"),
        ("photo of x", "photo of x, smiling", "photo of x, smiling"),
        ("photo\nof   x", "smiling", "photo of x, smiling"),
        (" , ", "smiling", "smiling"),
    ],
)
def test_the_prefix_is_joined_once_with_a_comma(
    data_root: Path, write_file: Write, prefix: str, caption: str, expected: str
) -> None:
    write_file("prefix.txt", prefix)
    write_file("a.png", "")
    write_file("a.txt", caption)

    assert _captions(data_root) == {"a.png": expected}


def test_a_prefix_one_level_above_the_dataset_applies_but_not_two(tmp_path: Path) -> None:
    root = tmp_path / "far" / "near" / "data"
    _write(root / "a.png", "")
    _write(tmp_path / "far" / "prefix.txt", "too far")

    before = _captions(root)
    _write(tmp_path / "far" / "near" / "prefix.txt", "[folder] style")

    assert before == {"a.png": ""}
    assert _captions(root) == {"a.png": "data style"}
