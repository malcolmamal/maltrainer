"""Differential output preservation: the caption without the trigger, held to the base model's prediction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from train_krea2_lora import Krea2LoRATrainer

from .runs import prepared_lora, progress_lines, run_lora
from .world import Workspace, write_image

DOP = {"dop": True, "dop_class": "person"}


def _cached(ws: Workspace, **train: Any) -> Krea2LoRATrainer:
    trainer = Krea2LoRATrainer(ws.lora_config(train={**DOP, **train}))
    trainer.precache_dataset()
    return trainer


def test_dop_needs_the_class_word_and_the_trigger(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match=r"train\.dop needs train\.dop_class, the class word"):
        Krea2LoRATrainer(workspace.lora_config(train={"dop": True, "dop_class": "  "}))
    with pytest.raises(ValueError, match=r"train\.dop needs dataset\.trigger: DOP's captions"):
        Krea2LoRATrainer(workspace.lora_config(train=DOP, dataset={"trigger": ""}))


def test_the_dop_caption_swaps_the_trigger_for_the_class_and_both_are_cached(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)

    with caplog.at_level(logging.INFO):
        trainer = _cached(workspace, dop_multiplier=2.0, dop_every=3)

    assert set(trainer.dop_captions.values()) == {"a photo of person"}
    assert sorted(trainer.text_cache) == ["a photo of person", "a photo of zed"]
    assert trainer.run_metadata["dop"] == {"class": "person", "multiplier": 2.0, "every": 3, "captions": 2}
    assert "DOP: 2 of 2 captions name the trigger 'zed'" in caplog.text
    assert "(e.g. 'a photo of zed' -> 'a photo of person')" in caplog.text


def test_a_caption_without_the_trigger_gets_no_dop_pass_and_a_warning(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(1)
    write_image(workspace.data / "tree.png", "a lone tree")

    with caplog.at_level(logging.WARNING):
        trainer = _cached(workspace)

    assert [Path(key).name for key in trainer.dop_captions] == ["img0.png"]
    assert "DOP: 1 caption(s) don't contain the trigger 'zed' and get no DOP pass" in caplog.text


def test_dop_with_no_caption_naming_the_trigger_is_refused(workspace: Workspace) -> None:
    write_image(workspace.data / "tree.png", "a lone tree")

    with pytest.raises(ValueError, match=r"train\.dop: no caption contains the trigger 'zed'"):
        _cached(workspace)


def test_dop_adds_a_second_loss_that_starts_at_zero_and_leaves_the_adapters_switched_on(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    losses: list[float | None] = []
    original = Krea2LoRATrainer._dop_loss

    def spy(self: Krea2LoRATrainer, index: int, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor | None:
        result = original(self, index, noisy, sigma)
        losses.append(None if result is None else result.item())
        return result

    monkeypatch.setattr(Krea2LoRATrainer, "_dop_loss", spy)

    trainer = run_lora(workspace.lora_config(train={**DOP, "max_steps": 3}))

    assert losses[0] == 0.0 and all(loss is not None and loss > 0 for loss in losses[1:]) and len(losses) == 3
    assert trainer.last_dop_loss == losses[-1] and trainer.micro_step == 3
    assert trainer.lora_network._active == [True]
    assert all("DOP: " in line for line in progress_lines(workspace))


def test_dop_every_runs_the_pass_on_every_nth_micro_step_only(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    losses: list[bool] = []
    original = Krea2LoRATrainer._dop_loss

    def spy(self: Krea2LoRATrainer, index: int, noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor | None:
        result = original(self, index, noisy, sigma)
        losses.append(result is not None)
        return result

    monkeypatch.setattr(Krea2LoRATrainer, "_dop_loss", spy)

    run_lora(workspace.lora_config(train={**DOP, "max_steps": 4, "dop_every": 2}))

    assert losses == [True, False, True, False]


def test_the_dop_loss_is_the_multiplied_distance_to_the_base_models_own_prediction(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared_lora(workspace.lora_config(train=DOP))
    for parameter in trainer.lora_network.trainable_parameters():
        torch.nn.init.normal_(parameter, std=0.5)
    _, noisy, sigma = trainer._training_step(0, [True])
    context = trainer.text_cache["a photo of person"].unsqueeze(0)
    with torch.no_grad(), trainer.lora_network.disabled():
        base = trainer.lora_network(noisy, sigma, context).float()
    with torch.no_grad():
        adapted = trainer.lora_network(noisy, sigma, context).float()

    plain = trainer._dop_loss(0, noisy, sigma)
    trainer.config["train"]["dop_multiplier"] = 2.5
    doubled = trainer._dop_loss(0, noisy, sigma)

    assert plain is not None and doubled is not None
    torch.testing.assert_close(plain, torch.nn.functional.mse_loss(adapted, base))
    torch.testing.assert_close(doubled, 2.5 * plain)
