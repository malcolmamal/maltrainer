"""Masked training: the mask beside a picture weights its loss; pictures without a usable one train as normal."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from maltrainer.data.masks import MASK_SUFFIX
from maltrainer.training.masked_loss import prior_loss, weighted_mse
from maltrainer.utils.lora_metadata import read_metadata

from .runs import prepared_subjects, run_subjects
from .world import Workspace

MASKED = {"masks": {"enabled": True, "floor": 0.0}}


def _mask(image: Path, white: tuple[int, int, int, int] | None, size: tuple[int, int] = (32, 32)) -> Path:
    """A mask beside ``image``: white inside ``white`` (a box), black elsewhere; ``None``: all black."""
    mask = Image.new("L", size, 0)
    if white is not None:
        mask.paste(255, white)
    path = image.with_name(image.stem + MASK_SUFFIX + ".png")
    mask.save(path)
    return path


def _left_half(ws: Workspace, index: int = 0) -> Path:
    return _mask(ws.data / f"img{index}.png", (0, 0, 16, 32))


def _randomised(trainer: Any) -> None:
    """Non-zero adapters, so the LoRA's prediction differs from the base model's."""
    for parameter in trainer.lora_network.trainable_parameters():
        torch.nn.init.normal_(parameter, std=0.3)


def test_masks_enabled_with_no_mask_beside_any_picture_is_refused(workspace: Workspace) -> None:
    workspace.add_images(2)

    with pytest.raises(ValueError, match=r"masks\.enabled is on but no image has a usable mask"):
        run_subjects(workspace.lora_config(**MASKED))


def test_mask_files_are_not_training_images_and_each_picture_gets_its_own_mask(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    _left_half(workspace)

    with caplog.at_level(logging.INFO):
        trainer = prepared_subjects(workspace.lora_config(**MASKED))

    assert sorted(Path(key).name for key, _ in trainer.samples) == ["img0.png", "img1.png"]
    assert list(trainer.mask_cache) == [str(workspace.data / "img0.png")]
    cached = trainer.mask_cache[str(workspace.data / "img0.png")]
    assert cached.shape == (1, 1, 4, 4) and cached[0, 0].tolist() == [[1.0, 1.0, 0.0, 0.0]] * 4
    assert "Masked training: 1 of 2 images have a mask (floor 0.00, prior 0.00)" in caplog.text


def test_a_mask_that_cannot_be_used_is_named_and_its_picture_trains_without_it(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)
    _left_half(workspace, 0)
    _mask(workspace.data / "img1.png", (0, 0, 8, 8), size=(32, 16))  # not its picture's shape
    _mask(workspace.data / "img2.png", None)  # all black

    with caplog.at_level(logging.WARNING):
        trainer = prepared_subjects(workspace.lora_config(**MASKED))

    assert len(trainer.mask_cache) == 1
    assert "img1-masklabel.png isn't the shape of its picture" in caplog.text
    assert "img2-masklabel.png is (almost) all black" in caplog.text
    assert "img1.png trains without it" in caplog.text


def _captured_step(trainer: Any) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """One step's loss, the network's outputs in call order, and the target ``noise - x0`` rebuilt from its input."""
    outputs: list[torch.Tensor] = []

    def hook(_module: Any, _args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        outputs.append(output.detach().float())

    handle = trainer.lora_network.register_forward_hook(hook)
    try:
        torch.manual_seed(5)
        loss, noisy, sigma = trainer._training_step(0, [True])
    finally:
        handle.remove()
    x0 = trainer.latent_cache[trainer.samples[0][0]].float()
    s = sigma.view(1, 1, 1, 1)
    return loss, outputs, (noisy.float() - (1 - s) * x0) / s - x0


def test_the_loss_is_the_error_weighted_by_the_mask_and_a_white_mask_changes_nothing(workspace: Workspace) -> None:
    workspace.add_images(1)
    _left_half(workspace)
    trainer = prepared_subjects(workspace.lora_config(masks={"enabled": True, "floor": 0.25}))
    _randomised(trainer)

    loss, outputs, target = _captured_step(trainer)
    weight = torch.tensor([[1.0, 1.0, 0.25, 0.25]] * 4).view(1, 1, 4, 4)

    assert len(outputs) == 1
    torch.testing.assert_close(loss, weighted_mse(outputs[0], target, weight))
    trainer.mask_cache[str(workspace.data / "img0.png")] = torch.ones(1, 1, 4, 4)
    white, *_ = _captured_step(trainer)
    trainer.masked = False
    plain, *_ = _captured_step(trainer)
    torch.testing.assert_close(white, plain)


def test_the_prior_holds_the_outside_to_the_base_models_prediction(workspace: Workspace) -> None:
    workspace.add_images(1)
    _left_half(workspace)
    trainer = prepared_subjects(workspace.lora_config(masks={"enabled": True, "floor": 0.0, "prior": 0.5}))
    _randomised(trainer)

    loss, outputs, target = _captured_step(trainer)
    base, pred = outputs
    weight = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * 4).view(1, 1, 4, 4)

    assert len(outputs) == 2 and not torch.equal(base, pred)
    torch.testing.assert_close(loss, weighted_mse(pred, target, weight) + prior_loss(pred, base, weight, 0.5))
    assert trainer.lora_network._active == [True]


def test_a_masked_run_with_a_prior_trains_saves_and_says_so_in_the_file(workspace: Workspace) -> None:
    workspace.add_images(2)
    _left_half(workspace, 0)
    _left_half(workspace, 1)

    run_subjects(workspace.lora_config(masks={"enabled": True, "floor": 0.1, "prior": 1.0}, train={"max_steps": 2}))

    saved = workspace.out / "k2.safetensors"
    assert load_file(str(saved))
    recorded = json.loads(read_metadata(saved)["maltrainer_masked_training"])
    assert recorded == {"floor": 0.1, "images_with_mask": 2, "prior": 1.0}


def test_a_dry_run_counts_pictures_and_not_their_masks(workspace: Workspace) -> None:
    workspace.add_images(2)
    _left_half(workspace)

    trainer = prepared_subjects(workspace.lora_config(**MASKED))

    assert trainer._dry_run_samples() == 2
