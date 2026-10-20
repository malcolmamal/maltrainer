"""Which files a finetune trains on and with what caption: hidden folders, captions, broken images, regularisation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

import train_moss_image_finetune as finetune

from .runs import cached_finetune, record_sample_order, run_finetune
from .world import Workspace


def _names(trainer: finetune.MOSSImageFinetuneTrainer) -> list[str]:
    return [Path(key).relative_to(Path(trainer.config["dataset"]["path"])).as_posix() for key, _ in trainer.samples]


def test_hidden_folders_are_never_trained_on(workspace: Workspace) -> None:
    workspace.add_images("p", 2)
    workspace.add_images(".trash", 2)
    workspace.add_images("p/.ipynb_checkpoints", 1)

    trainer = cached_finetune(workspace.config())

    assert _names(trainer) == ["p/img0.png", "p/img1.png"]


@pytest.mark.parametrize(
    ("trigger", "default", "expected"),
    [
        ("zed", "a photo of zed", "a photo of zed"),
        ("zed", "a photo", "zed a photo"),
        ("zed", "", "zed"),
        ("", "a photo", "a photo"),
        ("", "", ""),
    ],
)
def test_an_image_without_a_caption_file_gets_the_trigger_and_default_caption(
    workspace: Workspace, trigger: str, default: str, expected: str
) -> None:
    workspace.add_images("p", 1, caption=None)

    trainer = cached_finetune(workspace.config(dataset={"trigger": trigger, "default_caption": default}))

    assert [caption for _, caption in trainer.samples] == [expected]
    assert expected in trainer.text_cache


def test_a_caption_file_beats_the_default_and_a_prefix_file_is_put_in_front(workspace: Workspace) -> None:
    workspace.add_images("p", 1, caption="smiling")
    (workspace.data / "p" / "prefix.txt").write_text("photo of [folder]", encoding="utf-8")

    trainer = cached_finetune(workspace.config())

    assert [caption for _, caption in trainer.samples] == ["photo of p, smiling"]


def test_a_broken_image_is_skipped_with_a_warning_and_the_rest_trains(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2)
    (workspace.data / "p" / "broken.png").write_bytes(b"not an image")

    with caplog.at_level(logging.WARNING):
        trainer = run_finetune(workspace.config())

    assert _names(trainer) == ["p/img0.png", "p/img1.png"]
    assert "Skipping broken.png" in caplog.text
    assert trainer.global_step == 3


def test_a_dataset_where_every_image_is_broken_stops_before_training(workspace: Workspace) -> None:
    (workspace.data / "p").mkdir(parents=True)
    (workspace.data / "p" / "broken.png").write_bytes(b"not an image")

    with pytest.raises(ValueError, match=r"No usable images: every file in the dataset failed to load"):
        run_finetune(workspace.config())


def _regularised(ws: Workspace, **extra: Any) -> dict[str, Any]:
    ws.add_images("p", 2)
    ws.add_images(ws.root / "reg", 3, caption=None)
    return ws.config(dataset={"reg_path": str(ws.root / "reg"), "reg_default_caption": "a photo of a person"}, **extra)


def test_regularisation_images_join_the_samples_with_their_own_default_caption(workspace: Workspace) -> None:
    trainer = cached_finetune(_regularised(workspace, train={"reg_lr_multiplier": 0.3}))

    assert sorted(Path(key).name for key in trainer.reg_keys) == ["img0.png", "img1.png", "img2.png"]
    assert len(trainer.samples) == 5
    assert {caption for key, caption in trainer.samples if key in trainer.reg_keys} == {"a photo of a person"}
    assert trainer.reg_multiplier == 0.3


@pytest.mark.parametrize(("reg_repeats", "expected"), [(1, 5), (2, 8), (0, 2)])
def test_reg_repeats_sets_how_often_regularisation_images_appear(
    workspace: Workspace, reg_repeats: int, expected: int
) -> None:
    config = _regularised(workspace)
    config["dataset"]["reg_repeats"] = reg_repeats

    trainer = cached_finetune(config)

    assert (len(trainer._epoch_order()), trainer._epoch_size()) == (expected, expected)


@pytest.mark.parametrize("fused", [True, False])
def test_regularisation_images_step_at_the_reduced_size(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, fused: bool
) -> None:
    order = record_sample_order(monkeypatch)
    scales: list[float] = []
    original = finetune.set_step_scale

    def noting(optimizer: Any, scale: float) -> None:
        scales.append(scale)
        original(optimizer, scale)

    monkeypatch.setattr(finetune, "set_step_scale", noting)
    config = _regularised(workspace, train={"automagic_fused": fused, "max_steps": 5, "reg_lr_multiplier": 0.25})

    trainer = run_finetune(config)

    expected = [0.25 if trainer.samples[i][0] in trainer.reg_keys else 1.0 for i in order]
    assert scales == expected
    assert trainer.optimizer.lr_scale == scales[-1]


def test_an_empty_regularisation_folder_is_refused(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    (workspace.root / "reg").mkdir()

    with pytest.raises(ValueError, match=r"dataset\.reg_path .*reg has no images"):
        cached_finetune(workspace.config(dataset={"reg_path": str(workspace.root / "reg")}))
