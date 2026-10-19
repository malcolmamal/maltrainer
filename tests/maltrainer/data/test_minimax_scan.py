from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import pytest

from maltrainer.data.minimax_dataset import (
    MediaSource,
    _media_kind,
    count_media_files,
    scan_media_sources,
    summarize_kinds,
)

Write = Callable[[str, "str | bytes"], Path]


def _layout(write_file: Write) -> None:
    for rel in ["b.png", "a.JPG", "clips/c.mp4", "clips/d_mute.MOV", "voice/e.wav", "voice/f.opus"]:
        write_file(rel, "")
    for rel in ["notes.txt", "prefix.txt", "model.safetensors", "g.gif"]:
        write_file(rel, "")


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("a.png", "image"),
        ("a.JPEG", "image"),
        ("a.webp", "image"),
        ("a.mkv", "clip"),
        ("a.M4V", "clip"),
        ("a.flac", "audio"),
        ("a.aac", "audio"),
        ("a.txt", None),
        ("a.gif", None),
        ("png", None),
    ],
)
def test_the_extension_decides_the_media_kind(name: str, kind: str | None) -> None:
    assert _media_kind(Path(name)) == kind


@pytest.mark.parametrize(("name", "muted"), [("a_mute.mp4", True), ("A_MUTE.mp4", True), ("mute.mp4", False)])
def test_a_stem_ending_in_mute_marks_a_muted_source(name: str, muted: bool) -> None:
    assert MediaSource(Path(name), "clip", "").muted is muted


def test_every_supported_file_is_found_sorted_by_path(data_root: Path, write_file: Write) -> None:
    _layout(write_file)

    sources = scan_media_sources(data_root)

    assert [(s.path.relative_to(data_root).as_posix(), s.kind) for s in sources] == [
        ("a.JPG", "image"),
        ("b.png", "image"),
        ("clips/c.mp4", "clip"),
        ("clips/d_mute.MOV", "clip"),
        ("voice/e.wav", "audio"),
        ("voice/f.opus", "audio"),
    ]


@pytest.mark.parametrize(("kinds", "count"), [(["image"], 2), (["clip", "audio"], 4), ([], 0)])
def test_only_the_requested_kinds_are_scanned_and_counted(
    data_root: Path, write_file: Write, kinds: list[str], count: int
) -> None:
    _layout(write_file)

    sources = scan_media_sources(data_root, kinds=kinds)

    assert {s.kind for s in sources} == set(kinds)
    assert len(sources) == count_media_files(data_root, kinds) == count


def test_all_kinds_are_counted_by_default(data_root: Path, write_file: Write) -> None:
    _layout(write_file)

    assert count_media_files(data_root) == count_media_files(str(data_root)) == 6


def test_an_unknown_kind_is_an_error(data_root: Path) -> None:
    with pytest.raises(ValueError, match=r"Unknown dataset media kind\(s\) \['video'\]"):
        scan_media_sources(data_root, kinds=["image", "video"])


def test_a_missing_folder_has_no_sources_and_counts_zero(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="maltrainer.data.minimax_dataset"):
        sources = scan_media_sources(tmp_path / "missing")

    assert sources == []
    assert "does not exist" in caplog.text
    assert count_media_files(tmp_path / "missing") == 0


@pytest.mark.parametrize(
    ("caption", "trigger", "default", "expected"),
    [
        ("  a cat \n", None, "", "a cat"),
        ("a cat", "sks", "a photo", "a cat"),
        (None, "sks", "a photo", "sks a photo"),
        (None, " sks ", " a photo of sks ", "a photo of sks"),
        (None, "sks", "", "sks"),
        (None, None, "a photo", "a photo"),
        (None, None, "", ""),
    ],
)
def test_the_sidecar_caption_wins_over_the_trigger_and_default_caption(
    data_root: Path, write_file: Write, caption: str | None, trigger: str | None, default: str, expected: str
) -> None:
    write_file("a.png", "")
    if caption is not None:
        write_file("a.txt", caption)

    (source,) = scan_media_sources(data_root, trigger_token=trigger, default_caption=default)

    assert source.caption == expected


def test_an_unreadable_caption_falls_back_with_a_warning(
    data_root: Path, write_file: Write, caplog: pytest.LogCaptureFixture
) -> None:
    write_file("a.png", "")
    write_file("a.txt", b"\xff\xfe\xfa")

    with caplog.at_level(logging.WARNING, logger="maltrainer.data.minimax_dataset"):
        (source,) = scan_media_sources(data_root, trigger_token="sks")

    assert source.caption == "sks"
    assert "Error reading" in caplog.text


def test_summaries_count_every_kind_with_plurals() -> None:
    sources = [MediaSource(Path(n), k, "") for n, k in [("a.png", "image"), ("b.mp4", "clip"), ("c.mp4", "clip")]]

    assert summarize_kinds(sources) == "1 image, 2 clips, 0 audios"
    assert summarize_kinds([]) == "0 images, 0 clips, 0 audios"


def test_images_in_hidden_folders_are_neither_scanned_nor_counted(write_file: Write) -> None:
    for rel in ["a.png", "clips/b.mp4", ".ipynb_checkpoints/a-checkpoint.png", "clips/.trash/c.mp4"]:
        write_file(rel, "")
    root = write_file("a.png", "").parent

    assert [source.path.name for source in scan_media_sources(root)] == ["a.png", "b.mp4"]
    assert count_media_files(root) == 2
