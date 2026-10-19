from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pytest

from maltrainer.finetune.notes import names_its_folder, step_notes


def _folder(root: Path, name: str, captions: Dict[str, str], prefix: Optional[str] = None) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    for stem, caption in captions.items():
        (folder / f"{stem}.jpg").write_bytes(b"x")
        if caption is not None:
            (folder / f"{stem}.txt").write_text(caption, encoding="utf-8")
    if prefix is not None:
        (folder / "prefix.txt").write_text(prefix, encoding="utf-8")
    return folder


def _config(root: Path, prior: Optional[dict] = None, **dataset: object) -> dict:
    config: dict = {"dataset": {"path": str(root), **dataset}}
    if prior is not None:
        config["prior"] = prior
    return config


def test_a_fixed_prefix_makes_a_folder_a_concept_and_a_folder_token_makes_it_a_person(tmp_path: Path) -> None:
    concept = _folder(tmp_path, "eyes", {"a": "closeup of eyes"}, prefix="extreme close-up photo of human eyes")
    person = _folder(tmp_path, "anna", {"a": "smiling"}, prefix="photo of [folder], ")

    assert not names_its_folder(str(tmp_path), concept, ["closeup of eyes"])
    assert names_its_folder(str(tmp_path), person, ["smiling"])


def test_the_nearest_prefix_decides_even_when_it_sits_above_the_folder(tmp_path: Path) -> None:
    (tmp_path / "prefix.txt").write_text("photo of [folder]", encoding="utf-8")
    nested = _folder(tmp_path, "people/anna", {"a": "x"})

    assert names_its_folder(str(tmp_path), nested, ["x"])


def test_without_a_prefix_a_caption_must_spell_the_folder_name_as_a_word(tmp_path: Path) -> None:
    folder = _folder(tmp_path, "anna", {"a": "x"})

    assert names_its_folder(str(tmp_path), folder, ["photo of Anna smiling"])
    assert not names_its_folder(str(tmp_path), folder, ["photo of hanna smiling", "annabelle"])


def test_a_folder_repeats_entry_that_matches_no_folder_is_named(tmp_path: Path) -> None:
    _folder(tmp_path, "anna", {"a": "photo of anna"})

    notes = step_notes(_config(tmp_path, folder_repeats={"annna": 2, "anna": 1}))

    assert notes == ["dataset.folder_repeats names annna, which matches no folder (no effect)"]


def test_images_without_any_caption_are_named_with_their_folder(tmp_path: Path) -> None:
    _folder(tmp_path, "anna", {"a": "photo of anna"})
    _folder(tmp_path, "blank", {"a": None, "b": "has one"})

    (note,) = step_notes(_config(tmp_path))

    assert "images with no caption at all in blank:" in note
    assert "anna" not in note


def test_nothing_is_said_about_the_prior_when_its_weight_is_zero(tmp_path: Path) -> None:
    _folder(tmp_path, "anna", {"a": "photo of anna"})

    assert step_notes(_config(tmp_path, prior={"weight": 0})) == []
    assert step_notes(_config(tmp_path, prior={})) == []


def test_a_prior_without_classes_reports_the_config_error(tmp_path: Path) -> None:
    _folder(tmp_path, "anna", {"a": "photo of anna"})

    (note,) = step_notes(_config(tmp_path, prior={"weight": 0.5}))

    assert note.startswith("prior.weight > 0 needs prior.classes")


def test_a_person_without_a_prior_class_and_a_class_without_a_person_are_both_named(tmp_path: Path) -> None:
    _folder(tmp_path, "anna", {"a": "photo of anna"})
    _folder(tmp_path, "bob", {"a": "photo of bob"})
    _folder(tmp_path, "eyes", {"a": "closeup"}, prefix="close-up of human eyes")

    notes = step_notes(_config(tmp_path, prior={"weight": 1.0, "classes": {"anna": "woman", "zed": "man"}}))

    assert notes == [
        "prior.classes has no entry for bob: add each as <folder>: woman / man, or those people train without prior"
        " preservation",
        "prior.classes names zed, which no folder here names (harmless)",
    ]


@pytest.mark.parametrize("classes", [{"ANNA": "woman"}, {"anna": "woman"}])
def test_class_names_match_folders_ignoring_case(tmp_path: Path, classes: Dict[str, str]) -> None:
    _folder(tmp_path, "anna", {"a": "photo of anna"})

    assert step_notes(_config(tmp_path, prior={"weight": 1.0, "classes": classes})) == []


def test_a_single_trigger_with_a_class_word_is_a_prior_class(tmp_path: Path) -> None:
    _folder(tmp_path, "anna", {"a": "photo of anna"})

    config = _config(tmp_path, prior={"weight": 1.0, "class_word": "woman"}, trigger="anna")

    assert step_notes(config) == []
