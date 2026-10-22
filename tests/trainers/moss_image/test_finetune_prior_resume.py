"""A resumed run keeps its recorded prior, and records images added since from the original weights."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch

import train_moss_image_finetune as finetune
from maltrainer.models.moss_image.loading import load_moss_image_transformer

from .fakes import CPU
from .runs import base_prediction, run_finetune, same_values
from .test_finetune_prior import PRIOR
from .world import Workspace, base_weights, named_weights


def _first_run(ws: Workspace, **sections: Any) -> finetune.MOSSImageFinetuneTrainer:
    ws.add_images("old", 2, caption="a photo of zed")
    return run_finetune(ws.config(train={"max_steps": 2}, **sections))


def _resume_config(ws: Workspace, **sections: Any) -> dict[str, Any]:
    return ws.config(train={"max_steps": 4, "resume_from": str(ws.out / "ft_state")}, **sections)


def test_images_added_on_a_resumed_run_are_recorded_from_the_original_weights_then_the_trained_ones_return(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _first_run(workspace, prior=PRIOR)
    workspace.add_images("new", 2, caption="a photo of zed")
    trainer = finetune.MOSSImageFinetuneTrainer(_resume_config(workspace, prior=PRIOR))
    assert trainer.prior is not None
    seen: dict[str, Any] = {}
    build, setup = trainer.prior.build, trainer._setup_prior

    def build_spy(items: Any, *args: Any) -> None:
        seen["during"] = named_weights(trainer.transformer)
        items = list(items)
        seen["recorded"] = [rel for rel, _, _ in items]
        build(items, *args)

    def setup_spy() -> None:
        seen["before"] = named_weights(trainer.transformer)
        setup()
        seen["after"] = named_weights(trainer.transformer)

    monkeypatch.setattr(trainer.prior, "build", build_spy)
    monkeypatch.setattr(trainer, "_setup_prior", setup_spy)
    trainer.run()

    assert seen["recorded"] == ["new/img0.png", "new/img1.png"]
    assert same_values(seen["during"], base_weights(workspace))
    assert same_values(seen["before"], named_weights(first.transformer))
    assert same_values(seen["after"], seen["before"])
    assert trainer.prior is not None and first.prior is not None
    assert all(torch.equal(trainer.prior.targets[rel], first.prior.targets[rel]) for rel in first.prior.targets)


def test_the_images_added_on_resume_get_the_original_models_predictions(workspace: Workspace) -> None:
    _first_run(workspace, prior=PRIOR)
    workspace.add_images("new", 1, caption="a photo of zed")
    trainer = run_finetune(_resume_config(workspace, prior=PRIOR))
    base = load_moss_image_transformer(workspace.layout, CPU, torch.bfloat16)[0]
    prior = trainer.prior
    assert prior is not None

    x0 = trainer.latent_cache[str(workspace.data / "new" / "img0.png")]
    context = trainer.text_cache["a photo of person"]

    assert sorted(prior.targets) == ["new/img0.png", "old/img0.png", "old/img1.png"]
    for i in range(prior.samples_per_image):
        expected = base_prediction(base, x0, prior.sigmas["new/img0.png"][i], prior.seeds["new/img0.png"][i], context)
        assert torch.equal(prior.targets["new/img0.png"][i], expected.to(torch.bfloat16))


def test_a_changed_samples_per_image_applies_to_new_images_only_and_is_reported(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    _first_run(workspace, prior=PRIOR)
    workspace.add_images("new", 1, caption="a photo of zed")

    with caplog.at_level(logging.WARNING):
        trainer = run_finetune(_resume_config(workspace, prior={**PRIOR, "samples_per_image": 3}))

    assert trainer.prior is not None
    assert [len(trainer.prior.targets[rel]) for rel in sorted(trainer.prior.targets)] == [3, 2, 2]
    assert "has 2 inputs per image; new images get 3" in caplog.text


def test_resuming_with_a_prior_the_saved_run_never_had_is_refused(workspace: Workspace) -> None:
    _first_run(workspace)

    with pytest.raises(FileNotFoundError, match=r"prior\.weight > 0, but .*prior\.safetensors is missing"):
        run_finetune(_resume_config(workspace, prior=PRIOR))


def test_resuming_with_the_prior_switched_off_just_continues(workspace: Workspace) -> None:
    _first_run(workspace, prior=PRIOR)

    trainer = run_finetune(_resume_config(workspace))

    assert (trainer.global_step, trainer.prior) == (4, None)
