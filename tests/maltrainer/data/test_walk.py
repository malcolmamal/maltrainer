from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from maltrainer.data.walk import count_files, walk_visible

IMAGE_EXTENSIONS = {".png", ".jpg"}
HIDDEN = [".ipynb_checkpoints/a-checkpoint.png", "people/.trash/b.png", ".maltrainer_cache/latents/c.safetensors"]


def _visible_tree(write_file: Callable[[str, str], Path]) -> None:
    for rel in ["a.png", "B.JPG", "people/ben/c.png", "people/notes.txt", *HIDDEN]:
        write_file(rel, "")


def test_hidden_folders_are_not_entered_at_any_depth(data_root: Path, write_file: Callable[[str, str], Path]) -> None:
    _visible_tree(write_file)

    folders = sorted(Path(folder).relative_to(data_root).as_posix() for folder, _, _ in walk_visible(data_root))

    assert folders == [".", "people", "people/ben"]


def test_the_listed_subfolders_never_include_a_hidden_one(
    data_root: Path, write_file: Callable[[str, str], Path]
) -> None:
    _visible_tree(write_file)

    listed = {
        Path(folder).relative_to(data_root).as_posix(): sorted(dirs) for folder, dirs, _ in walk_visible(data_root)
    }

    assert listed == {".": ["people"], "people": ["ben"], "people/ben": []}


def test_hidden_files_are_still_listed_in_a_visible_folder(
    data_root: Path, write_file: Callable[[str, str], Path]
) -> None:
    write_file(".hidden.png", "")

    ((_, _, files),) = list(walk_visible(data_root))

    assert files == [".hidden.png"]


def test_a_hidden_root_is_walked_like_any_other(tmp_path: Path) -> None:
    root = tmp_path / ".cache" / "dataset"
    (root / "people").mkdir(parents=True)
    (root / "people" / "a.png").write_bytes(b"")
    (root / ".trash").mkdir()
    (root / ".trash" / "b.png").write_bytes(b"")

    assert [Path(folder).name for folder, _, _ in walk_visible(root)] == ["dataset", "people"]
    assert count_files(root, IMAGE_EXTENSIONS) == 1


def test_a_missing_root_is_an_empty_walk(tmp_path: Path) -> None:
    assert list(walk_visible(tmp_path / "absent")) == []
    assert count_files(tmp_path / "absent", IMAGE_EXTENSIONS) == 0


def test_files_are_counted_by_extension_in_any_case_outside_hidden_folders(
    data_root: Path, write_file: Callable[[str, str], Path]
) -> None:
    _visible_tree(write_file)

    assert count_files(data_root, IMAGE_EXTENSIONS) == 3
    assert count_files(data_root, {".txt"}) == 1
    assert count_files(str(data_root), set()) == 0
