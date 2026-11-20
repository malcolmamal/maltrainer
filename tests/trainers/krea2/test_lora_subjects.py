"""Up to four subjects in one LoRA: joined triggers, DOP for each, balance and the side-by-sides at a higher rate."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from maltrainer.trainers.krea2.lora_subjects import Krea2SubjectsLoRATrainer, with_subject_defaults
from maltrainer.utils.config import load_config

from .runs import prepared_subjects, run_subjects
from .subjects_world import sbs, subject, two
from .world import Workspace, write_image


def test_the_first_subject_stands_in_for_the_dataset_and_dop_starts_at_fizgigs_setting() -> None:
    config = {"isolation": {"subjects": [{"path": "p", "trigger": "t"}]}, "train": {"dop_every": 5}}

    prepared = with_subject_defaults(config)

    assert prepared["dataset"] == {"path": "p", "trigger": "t"}
    assert prepared["train"] == {"dop_every": 5, "dop": True, "dop_multiplier": 2.0, "dop_class": "subject"}
    assert "dataset" not in config and with_subject_defaults({"a": 1}) == {"a": 1}
    assert with_subject_defaults({"isolation": {"subjects": [{}]}, "train": {"dop": False}})["train"]["dop"] is False


def test_every_subject_trains_with_its_joined_word_and_dop_swaps_each_back_to_its_class(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    config = workspace.lora_config(isolation=two(workspace), train={"max_steps": 4})

    with caplog.at_level(logging.INFO):
        trainer = run_subjects(config)

    assert {caption for _, caption in trainer.samples} == {"zwxemwoman smiling", "qrxtman standing"}
    assert set(trainer.dop_captions.values()) == {"a woman smiling", "a man standing"}
    assert {"a woman smiling", "a man standing", "zwxemwoman smiling"} <= set(trainer.text_cache)
    dop = trainer.run_metadata["dop"]
    assert dop["subjects"] == [
        {"trigger": "zwxem", "word": "zwxemwoman", "class": "a woman"},
        {"trigger": "qrxt", "word": "qrxtman", "class": "a man"},
    ]
    assert (dop["multiplier"], dop["every"], dop["captions"]) == (2.0, 2, 4)
    assert "DOP: 4 of 4 captions name a subject" in caplog.text
    assert (workspace.out / "k2.safetensors").exists()


def test_a_smaller_folder_repeats_so_each_subject_trains_as_often(workspace: Workspace) -> None:
    trainer = prepared_subjects(workspace.lora_config(isolation=two(workspace, first=1, second=3)))

    owners = [Path(key).name[0] for key, _ in trainer.samples]

    assert (owners.count("a"), owners.count("b")) == (3, 3)
    assert trainer.run_metadata["isolation_balance"] == [3, 1]
    unbalanced = prepared_subjects(workspace.lora_config(isolation=two(workspace, balance=False)))
    assert len(unbalanced.samples) == 4


def test_captions_that_name_no_subject_get_no_dop_pass_and_none_at_all_is_refused(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    isolation = two(workspace, first=1, second=1)
    write_image(workspace.root / "a" / "extra.png", "a lone tree")
    with caplog.at_level(logging.WARNING):
        trainer = prepared_subjects(workspace.lora_config(isolation=isolation))
    assert len(trainer.dop_captions) == 2 and "1 caption(s) name no subject" in caplog.text

    stylist = {"subjects": [subject(workspace, "s", "papercut", "a tree", 1, kind="style")]}
    with pytest.raises(ValueError, match="no caption contains any subject's trigger"):
        prepared_subjects(workspace.lora_config(isolation=stylist))


def test_without_subjects_the_single_trigger_run_is_the_ordinary_one(workspace: Workspace) -> None:
    workspace.add_images(2)

    trainer = prepared_subjects(workspace.lora_config(train={"dop": True, "dop_class": "person"}))

    assert set(trainer.dop_captions.values()) == {"a photo of person"}
    assert trainer._dry_run_samples() == 2 and trainer.subjects == []


def test_a_dry_run_counts_the_epoch_as_the_run_will_have_it(workspace: Workspace) -> None:
    isolation = two(workspace, side_by_side=sbs(workspace))
    config = workspace.lora_config(isolation=isolation, dataset={"repeats": 2})

    assert Krea2SubjectsLoRATrainer(config)._dry_run_samples() == (3 + 3 + 2 * 2) * 2  # balanced, + sbs, x repeats
    isolation["balance"], isolation["side_by_side"]["enabled"] = False, False
    assert Krea2SubjectsLoRATrainer(config)._dry_run_samples() == (1 + 3) * 2


def test_the_example_template_is_a_config_the_trainer_accepts() -> None:
    config = load_config(str(Path(__file__).parents[3] / "templates" / "examples" / "train_krea2_lora_isolation.yaml"))

    prepared = with_subject_defaults(config)

    assert [s["trigger"] for s in prepared["isolation"]["subjects"]] == ["zwxem", "qrxt"]
    assert (
        prepared["train"]["dop"],
        prepared["masks"]["enabled"],
        prepared["isolation"]["side_by_side"]["enabled"],
    ) == (
        True,
        False,
        False,
    )
