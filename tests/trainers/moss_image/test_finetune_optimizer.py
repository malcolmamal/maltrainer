"""The finetune's Automagic3: its groups, its fused step inside backward, accumulation and its refusals."""

from __future__ import annotations

from typing import Any, cast

import pytest

import train_moss_image_finetune as finetune

from .runs import changed, prepared_finetune, run_finetune
from .test_finetune_trainable import BLOCK_LINEARS
from .world import Workspace, named_weights


def _after_one_backward(ws: Workspace, **train: object) -> tuple[finetune.MOSSImageFinetuneTrainer, dict]:
    ws.add_images("p", 2)
    trainer = prepared_finetune(ws.config(train=train))
    before = named_weights(trainer.transformer)

    trainer._training_step(0).backward()

    return trainer, before


def test_a_fused_optimizer_steps_every_parameter_inside_backward_and_frees_its_gradient(workspace: Workspace) -> None:
    trainer, before = _after_one_backward(workspace)

    assert trainer.optimizer.fused
    assert changed(before, named_weights(trainer.transformer)) == BLOCK_LINEARS
    assert all(parameter.grad is None for _, parameter in trainer.named_trainable)


def test_an_unfused_optimizer_holds_the_gradients_until_step(workspace: Workspace) -> None:
    trainer, before = _after_one_backward(workspace, automagic_fused=False)

    assert changed(before, named_weights(trainer.transformer)) == set()
    assert all(hasattr(parameter, "_accum_grad") for _, parameter in trainer.named_trainable)

    trainer.optimizer.step()

    assert changed(before, named_weights(trainer.transformer)) == BLOCK_LINEARS
    assert all(trainer.optimizer.state[parameter]["step"] == 1 for _, parameter in trainer.named_trainable)


def test_fused_steps_with_gradient_accumulation_are_refused(workspace: Workspace) -> None:
    workspace.add_images("p", 2)
    config = workspace.config(train={"gradient_accumulation_steps": 2})

    with pytest.raises(
        ValueError, match=r"automagic_fused steps inside backward, so gradient_accumulation_steps must be 1"
    ):
        prepared_finetune(config)


def test_unfused_accumulation_takes_one_optimizer_step_per_accumulated_micro_steps(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images("p", 4)
    seen: list[int] = []
    original = finetune.MOSSImageFinetuneTrainer._training_step

    def counting(self: finetune.MOSSImageFinetuneTrainer, index: int) -> object:
        seen.append(index)
        return original(self, index)

    monkeypatch.setattr(finetune.MOSSImageFinetuneTrainer, "_training_step", counting)

    trainer = run_finetune(
        workspace.config(train={"gradient_accumulation_steps": 2, "automagic_fused": False, "max_steps": 2})
    )

    assert (trainer.global_step, len(seen), sorted(seen)) == (2, 4, [0, 1, 2, 3])
    assert all(trainer.optimizer.state[parameter]["step"] == 2 for _, parameter in trainer.named_trainable)


@pytest.mark.parametrize("name", ["automagic3", "Automagic", "automagic_v3"])
def test_the_automagic_spellings_all_select_automagic3(workspace: Workspace, name: str) -> None:
    workspace.add_images("p", 1)

    trainer = prepared_finetune(workspace.config(train={"optimizer": name}))

    assert cast(dict[str, Any], trainer.run_metadata["optimizer_resolved"])["name"] == "automagic3"


@pytest.mark.parametrize("name", ["adamw", "prodigy", "adamw8bit"])
def test_any_other_optimizer_is_refused(workspace: Workspace, name: str) -> None:
    workspace.add_images("p", 1)

    with pytest.raises(ValueError, match=rf"supports automagic3 only \(got '{name}'\)"):
        prepared_finetune(workspace.config(train={"optimizer": name}))


def test_the_resolved_optimizer_settings_come_from_the_train_section(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    train = {
        "learning_rate": 2e-4,
        "automagic_min_lr": 1e-6,
        "automagic_max_lr": 5e-4,
        "automagic_beta2": 0.99,
        "automagic_eps": 1e-20,
        "automagic_clip_threshold": 0.5,
        "weight_decay": 0.01,
        "automagic_polarity_history": 4,
    }

    trainer = prepared_finetune(workspace.config(train=train))

    resolved = trainer.run_metadata["optimizer_resolved"]
    assert resolved == {
        "name": "automagic3",
        "groups": ["attn", "mlp"],
        "lr": 2e-4,
        "min_lr": 1e-6,
        "max_lr": 5e-4,
        "beta2": 0.99,
        "eps": 1e-20,
        "clip_threshold": 0.5,
        "weight_decay": 0.01,
        "polarity_history": 4,
        "fused": True,
    }
    assert [group["lr"] for group in trainer.optimizer.param_groups] == [2e-4, 2e-4]


def test_the_groups_split_by_module_family_or_collapse_into_one(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    everything = {"trainable_prefixes": [], "linears_only": False}

    split = prepared_finetune(workspace.config(finetune=everything))
    single = prepared_finetune(workspace.config(finetune=everything, train={"automagic_param_groups": False}))

    assert sorted(group["name"] for group in split.optimizer.param_groups) == ["attn", "mlp", "norm", "other"]
    assert [group["name"] for group in single.optimizer.param_groups] == ["all"]
    assert sum(len(g["params"]) for g in split.optimizer.param_groups) == len(
        single.optimizer.param_groups[0]["params"]
    )
