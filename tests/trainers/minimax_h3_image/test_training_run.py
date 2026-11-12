"""A whole H3 Image run on the tiny world: stills only, the checkpoint's own levels, base untouched, files saved."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from safetensors.torch import load_file

import train_minimax_h3_image_lora as image
from maltrainer.models.minimax_h3_image.grid import PDD_SIGMAS
from maltrainer.utils.lora_metadata import read_metadata
from tests.trainers.minimax_h3.world import Workspace, named_weights, same_tensors

from .conftest import image_state

FINAL = "minimax_h3_image_lora_step_final.safetensors"


def _run(ws: Workspace, images: int = 2, **sections: Any) -> image.MiniMaxH3ImageTrainer:
    ws.add_images(images)
    trainer = image.MiniMaxH3ImageTrainer(ws.config(**sections))
    trainer.run()
    return trainer


def test_only_the_adapters_train_and_the_image_base_stays_bit_identical(workspace: Workspace) -> None:
    trainer = _run(workspace)

    assert same_tensors(named_weights(trainer.transformer), image_state())
    saved = load_file(str(workspace.out / FINAL))
    assert all(torch.count_nonzero(saved[k]) > 0 for k in saved if k.endswith("lora_B.weight"))


def test_the_checkpoint_is_labelled_as_h3_image_and_records_its_levels(workspace: Workspace) -> None:
    _run(workspace)

    meta = read_metadata(workspace.out / FINAL)
    assert meta["maltrainer_architecture"] == "minimax_h3_image"
    assert meta["maltrainer_trainer"] == "train_minimax_h3_image_lora.py"
    assert meta["ss_base_model_version"] == "minimax_h3_image"
    assert meta["maltrainer_noise_levels"] == str(list(PDD_SIGMAS[:-1]))


def test_clips_and_voices_in_the_dataset_folder_are_ignored(workspace: Workspace) -> None:
    workspace.add_clip()
    workspace.add_voice()

    trainer = _run(workspace, images=2, dataset={"media": ["image", "clip", "audio"]})

    assert [sample.kind for sample in trainer.samples] == ["image", "image"]


def test_every_step_runs_at_one_of_the_checkpoints_levels(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[float, int]] = []
    original = image.MiniMaxH3ImageTrainer._forward_image

    def spy(self: Any, patches: Any, text: Any, sigma: float, layout: Any, level: int) -> Any:
        seen.append((sigma, level))
        return original(self, patches, text, sigma, layout, level)

    monkeypatch.setattr(image.MiniMaxH3ImageTrainer, "_forward_image", spy)

    _run(workspace, images=3, train={"num_epochs": 4})

    assert len(seen) == 12 and all(PDD_SIGMAS[level] == sigma for sigma, level in seen)


def test_likeness_optimization_moves_steps_to_the_clean_end(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    levels: list[int] = []
    original = image.MiniMaxH3ImageTrainer._draw_level

    def spy(self: image.MiniMaxH3ImageTrainer) -> int:
        levels.append(original(self))
        return levels[-1]

    monkeypatch.setattr(image.MiniMaxH3ImageTrainer, "_draw_level", spy)

    _run(workspace, images=4, train={"num_epochs": 20, "likeness_opt": True})

    assert levels.count(5) > len(levels) * 0.5  # level 5 is sigma 0.232, the only one under 0.5


def test_the_epoch_follows_the_steps_taken(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = image.MiniMaxH3ImageTrainer(workspace.config())
    monkeypatch.setattr(trainer, "_epoch_order", lambda: [0, 1, 2])
    assert trainer._epoch() == 1
    trainer._steps_taken = 3
    assert trainer._epoch() == 2


def test_a_batch_above_one_is_refused(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="batch_size 1"):
        image.MiniMaxH3ImageTrainer(workspace.config(train={"batch_size": 2}))


def test_a_dry_run_writes_records_named_for_h3_image(workspace: Workspace) -> None:
    workspace.add_images(2)
    trainer = image.MiniMaxH3ImageTrainer(workspace.config(), dry_run=True, dry_run_step_time=0.0)

    trainer.run()

    assert (workspace.out / FINAL).exists()
