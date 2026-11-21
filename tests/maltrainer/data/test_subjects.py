"""The subjects of a multi-subject LoRA, and the side-by-sides built from them."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from maltrainer.data.composites import FACECROP, PREFIX, _picks, build_composites, load_strangers
from maltrainer.data.subjects import balanced_repeats, load_subjects


def _folder(root: Path, name: str, captions: list[str], size: tuple[int, int] = (16, 8)) -> Path:
    folder = root / name
    folder.mkdir()
    for i, caption in enumerate(captions):
        Image.new("RGB", size, (i * 20, 0, 0)).save(folder / f"p{i}.png")
        (folder / f"p{i}.txt").write_text(caption, encoding="utf-8")
    return folder


def _isolation(root: Path, **extra: Any) -> dict[str, Any]:
    woman = _folder(root, "a", ["zwxem, a woman smiling", "zwxem, a woman at a cafe"])
    man = _folder(root, "b", ["qrxt, a man standing"])
    return {
        "subjects": [{"path": str(woman), "trigger": "zwxem"}, {"path": str(man), "trigger": "qrxt"}],
        **extra,
    }


def test_each_subject_reads_its_captions_with_its_own_joined_word(tmp_path: Path) -> None:
    subjects = load_subjects(_isolation(tmp_path), "")

    assert [s.word for s in subjects] == ["zwxemwoman", "qrxtman"]
    assert [s.sources[0].caption for s in subjects] == ["zwxemwoman smiling", "qrxtman standing"]
    assert [s.words.phrase for s in subjects] == ["a woman", "a man"]


def test_the_joined_word_can_be_switched_off_or_written_by_hand(tmp_path: Path) -> None:
    plain = load_subjects(_isolation(tmp_path, fuse_triggers=False), "")
    assert (plain[0].word, plain[0].sources[0].caption) == ("zwxem", "zwxem, a woman smiling")

    folder = _folder(tmp_path, "c", ["zedwoman walking"])
    written = load_subjects({"subjects": [{"path": str(folder), "trigger": "zedwoman"}], "as_written": True}, "")
    assert (written[0].word, written[0].sources[0].caption) == ("zedwoman", "zedwoman walking")


def test_mask_files_are_not_images_of_a_subject(tmp_path: Path) -> None:
    isolation = _isolation(tmp_path)
    Image.new("L", (16, 8), 255).save(tmp_path / "a" / "p0-masklabel.png")

    assert len(load_subjects(isolation, "")[0].sources) == 2


@pytest.mark.parametrize(
    ("subjects", "message"),
    [
        ([], "needs 1 to 4 subjects, got 0"),
        ([{"path": "x", "trigger": "a"}] * 5, "needs 1 to 4 subjects, got 5"),
        ([{"path": "x"}], "needs a path and a trigger"),
    ],
)
def test_the_number_of_subjects_and_what_each_needs_are_checked(subjects: list[dict[str, str]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_subjects({"subjects": subjects}, "")


def test_a_subject_without_pictures_and_two_sharing_a_trigger_or_folder_are_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No images found for subject 'zed'"):
        load_subjects({"subjects": [{"path": str(empty), "trigger": "zed"}]}, "")

    folder = _folder(tmp_path, "same", ["zed, a man"])
    other = _folder(tmp_path, "other", ["zed, a man"])
    for entries in (
        [{"path": str(folder), "trigger": "zed"}, {"path": str(other), "trigger": "ZED"}],
        [{"path": str(folder), "trigger": "zed"}, {"path": str(folder), "trigger": "qrxt", "what": "man"}],
    ):
        with pytest.raises(ValueError, match="their own trigger word and their own folder"):
            load_subjects({"subjects": entries}, "")


def test_triggers_that_start_alike_are_warned_about(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    a = _folder(tmp_path, "a", ["zwxem, a man"])
    b = _folder(tmp_path, "b", ["zwxen, a man"])

    with caplog.at_level(logging.WARNING):
        load_subjects({"subjects": [{"path": str(a), "trigger": "zwxem"}, {"path": str(b), "trigger": "zwxen"}]}, "")

    assert "share a first letter" in caplog.text


def test_a_folder_without_captions_uses_the_default_caption(tmp_path: Path) -> None:
    folder = tmp_path / "d"
    folder.mkdir()
    Image.new("RGB", (8, 8)).save(folder / "x.png")
    entries = [{"path": str(folder), "trigger": "zed", "what": "dog", "default_caption": "a photo of zed"}]

    subjects = load_subjects({"subjects": entries}, "unused")

    assert subjects[0].sources[0].caption == "a photo of zeddog"


def test_smaller_folders_repeat_to_match_the_largest() -> None:
    assert balanced_repeats([36, 53]) == [1, 1]
    assert balanced_repeats([10, 50, 25]) == [5, 1, 2]


# ---------------------------------------------------------------------------- side-by-sides


def _strangers(root: Path, count: int = 3) -> Path:
    folder = root / "strangers"
    folder.mkdir()
    rows = []
    for i in range(count):
        Image.new("RGB", (12, 20), (0, 200, i * 30)).save(folder / f"s{i}.png")
        rows.append({"file": f"s{i}.png", "description": f"stranger number {i}"})
    (folder / "strangers.json").write_text(json.dumps(rows), encoding="utf-8")
    return folder


def test_the_strangers_need_a_list_and_their_pictures(tmp_path: Path) -> None:
    assert [s.description for s in load_strangers(_strangers(tmp_path))] == [f"stranger number {i}" for i in range(3)]
    with pytest.raises(ValueError, match=r"no usable strangers\.json"):
        load_strangers(tmp_path / "nowhere")
    (tmp_path / "strangers" / "s1.png").unlink()
    with pytest.raises(ValueError, match=r"files missing: \['s1.png'\]"):
        load_strangers(tmp_path / "strangers")


def test_a_side_by_side_holds_the_subject_and_a_stranger_in_one_wide_frame(tmp_path: Path) -> None:
    subject = load_subjects(_isolation(tmp_path), "")[0]
    strangers = load_strangers(_strangers(tmp_path))

    made = build_composites(subject, 0, strangers, tmp_path / "out", count=4, seed=1, panel=16)

    assert [s.path.name for s in made] == [f"{PREFIX}0_{k:02d}.png" for k in range(4)]
    for source in made:
        with Image.open(source.path) as sheet:
            assert sheet.size == (32, 16)
        assert source.caption.startswith("stranger number ") and "zwxemwoman on the" in source.caption
        assert source.caption.endswith("two people side by side, two close-up portraits")
    again = build_composites(subject, 0, strangers, tmp_path / "out", count=4, seed=1, panel=16)
    assert [s.caption for s in again] == [s.caption for s in made]  # the same seed makes the same pictures


def test_face_crops_are_picked_first_and_photos_cycle_when_there_are_few(tmp_path: Path) -> None:
    folder = _folder(tmp_path, "a", ["zed, a man"] * 3)
    (folder / "p1.png").rename(folder / f"{FACECROP}1.png")
    (folder / "p1.txt").rename(folder / f"{FACECROP}1.txt")
    subject = load_subjects({"subjects": [{"path": str(folder), "trigger": "zed"}]}, "")[0]
    picks = _picks([s.path for s in subject.sources], 5, random.Random(0))

    assert picks[0].name.startswith(FACECROP) and len(picks) == 5 and len(set(picks)) == 3


def test_a_photo_of_two_subjects_joins_both_triggers_whichever_folder_it_is_in(tmp_path: Path) -> None:
    isolation = _isolation(tmp_path)
    (tmp_path / "a" / "p0.txt").write_text("zwxem, a woman, and qrxt, a man, laughing at a table", encoding="utf-8")

    subjects = load_subjects(isolation, "")

    assert subjects[0].sources[0].caption == "zwxemwoman, and qrxtman, laughing at a table"
