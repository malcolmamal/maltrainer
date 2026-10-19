from __future__ import annotations

from pathlib import Path

from maltrainer.utils.step_planner import Group, folder_counts, report


def _touch(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_images_are_counted_per_folder_that_holds_them(tmp_path: Path) -> None:
    _touch(tmp_path, "top.png", "people/ann/1.jpg", "people/ann/2.JPEG", "people/ann/a.txt", "body/eyes/1.webp")

    assert folder_counts(tmp_path) == [(".", 1), ("body/eyes", 1), ("people/ann", 2)]


def test_hidden_folders_and_other_media_are_skipped(tmp_path: Path) -> None:
    _touch(tmp_path, ".maltrainer_cache/1.png", "clips/1.mp4", "clips/2.png")

    assert folder_counts(tmp_path) == [("clips", 1)]
    assert folder_counts(tmp_path, kinds=("image", "clip")) == [("clips", 2)]


def test_a_missing_root_has_no_folders(tmp_path: Path) -> None:
    assert folder_counts(tmp_path / "absent") == []


def test_a_group_totals_its_images_and_samples_with_per_folder_repeats() -> None:
    group = Group("Subjects", "/data", 2, [("people/ann", 3), ("people/ben", 4)], {"ben": 0.5})

    assert group.images == 7
    assert group.repeats_for("people/ann") == 2.0
    assert group.repeats_for("people/ben") == 0.5
    assert group.samples == 8


def test_the_report_suggests_max_steps_for_the_default_epochs() -> None:
    group = Group("Subjects", "/data", 2, [("a", 3), ("bb", 5)])

    text = report([group], {})

    assert text.splitlines() == [
        "Subjects: /data (8 images, 2 repeats = 16 samples per epoch)",
        "  a       3",
        "  bb      5",
        "",
        "One epoch: 16 samples = 16 steps",
        "Suggested for 22 epochs:  train.max_steps: 352",
        "Configured now:  max_steps 0, so num_epochs 10 = 160 steps",
    ]


def test_the_report_shows_each_folders_repeats_when_they_differ() -> None:
    group = Group("Subjects", "/data", 2, [("a", 3), ("b", 4)], {"b": 0.5})

    lines = report([group], {"num_epochs": 4}, epochs=1.5).splitlines()

    assert lines[:3] == [
        "Subjects: /data (7 images, per-folder repeats = 8 samples per epoch)",
        "  a      3  x 2    =    6",
        "  b      4  x 0.5  =    2",
    ]
    assert lines[5:] == [
        "Suggested for 1.5 epochs:  train.max_steps: 12",
        "Configured now:  max_steps 0, so num_epochs 4 = 32 steps",
    ]


def test_the_report_divides_by_gradient_accumulation_and_reads_the_configured_steps() -> None:
    groups = [Group("Subjects", "/data", 1, [("a", 10)]), Group("Regularisation", "/reg", 0, [])]

    lines = report(groups, {"gradient_accumulation_steps": 4, "max_steps": 30}, notes=["check the prior"]).splitlines()

    assert lines[2:] == [
        "Regularisation: /reg (0 images, 0 repeats = 0 samples per epoch)",
        "  (no images found)",
        "",
        "One epoch: 10 samples = 3 steps (gradient_accumulation_steps 4)",
        "Suggested for 22 epochs:  train.max_steps: 66",
        "Configured now:  max_steps 30 = 10.0 epochs",
        "Note: check the prior",
    ]


def test_an_empty_dataset_still_counts_one_step_per_epoch() -> None:
    lines = report([Group("Subjects", "/data", 1, [])], {"num_epochs": 0}).splitlines()

    assert lines[-3:] == [
        "One epoch: 0 samples = 1 steps",
        "Suggested for 22 epochs:  train.max_steps: 22",
        "Configured now:  max_steps 0, so num_epochs 1 = 1 steps",
    ]
