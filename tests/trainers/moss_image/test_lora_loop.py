"""The LoRA training loop: step planning, the step count, accumulation and clipping."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import train_moss_image_lora as lora
from maltrainer.training.steps import plan_training_steps

from .runs import prepared_lora, progress_lines, run_lora
from .world import Workspace


@pytest.mark.parametrize(
    ("samples", "train", "expected"),
    [
        (10, {}, (10, 10, 100)),
        (10, {"num_epochs": 3}, (10, 3, 30)),
        (10, {"max_steps": 25, "num_epochs": 99}, (10, 3, 25)),
        (10, {"gradient_accumulation_steps": 4, "num_epochs": 2}, (3, 2, 5)),
        (3, {"gradient_accumulation_steps": 2, "max_steps": 4}, (2, 3, 4)),
        (0, {}, (1, 10, 10)),
        (7, {"max_steps": None, "num_epochs": None, "gradient_accumulation_steps": None}, (7, 1, 7)),
        (7, {"max_steps": 0, "num_epochs": 0}, (7, 1, 7)),
    ],
)
def test_the_plan_gives_steps_per_epoch_epochs_and_total_steps(
    samples: int, train: dict[str, Any], expected: tuple[int, int, int]
) -> None:
    assert plan_training_steps(samples, train) == expected


def test_max_steps_stops_mid_epoch_and_the_log_counts_every_step(workspace: Workspace) -> None:
    workspace.add_images("p", 3)

    run_lora(workspace.lora_config(train={"max_steps": 5}))

    assert [line.split("Step: ")[1].split(",")[0] for line in progress_lines(workspace)] == [
        "1/5 [20%]",
        "2/5 [40%]",
        "3/5 [60%]",
        "4/5 [80%]",
        "5/5 [100%]",
    ]
    assert "Epoch: 2/2" in progress_lines(workspace)[-1]


def test_accumulation_over_a_sample_count_it_does_not_divide_still_takes_every_planned_step(
    workspace: Workspace,
) -> None:
    workspace.add_images("p", 3)

    run_lora(workspace.lora_config(train={"max_steps": 4, "gradient_accumulation_steps": 2}))

    lines = progress_lines(workspace)
    assert len(lines) == 4
    assert "Step: 4/4 [100%]" in lines[-1]


def test_repeats_make_the_epoch_longer_and_num_epochs_sets_the_total(workspace: Workspace) -> None:
    workspace.add_images("p", 3)

    trainer = run_lora(workspace.lora_config(dataset={"repeats": 2}, train={"max_steps": 0, "num_epochs": 2}))

    assert (trainer.run_metadata["steps_per_epoch"], trainer.run_metadata["planned_total_steps"]) == (6, 12)
    assert len(progress_lines(workspace)) == 12


def test_gradient_accumulation_takes_one_optimizer_step_for_several_micro_steps(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images("p", 4)
    seen: list[int] = []
    original = lora.MOSSImageLoRATrainer._training_step

    def counting(self: lora.MOSSImageLoRATrainer, index: int) -> torch.Tensor:
        seen.append(index)
        return original(self, index)

    monkeypatch.setattr(lora.MOSSImageLoRATrainer, "_training_step", counting)

    trainer = run_lora(workspace.lora_config(train={"gradient_accumulation_steps": 2, "max_steps": 2}))

    assert len(seen) == 4 and len(progress_lines(workspace)) == 2
    assert trainer.run_metadata["steps_per_epoch"] == 2


def test_gradients_are_clipped_to_max_grad_norm_before_each_step(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images("p", 2)
    trainer = prepared_lora(workspace.lora_config(train={"max_grad_norm": 0.25, "max_steps": 2}))
    norms: list[float] = []
    clip = trainer.accelerator.clip_grad_norm_

    def noting(params: Any, max_norm: float) -> Any:
        norms.append(max_norm)
        return clip(params, max_norm)

    monkeypatch.setattr(trainer.accelerator, "clip_grad_norm_", noting)

    trainer.train()

    assert norms == [0.25, 0.25]


def test_when_accumulation_cannot_reach_the_planned_steps_the_run_ends_after_its_epochs_and_still_saves(
    workspace: Workspace,
) -> None:
    workspace.add_images("p", 3)

    run_lora(workspace.lora_config(train={"gradient_accumulation_steps": 2, "max_steps": 4}))

    assert (workspace.out / "moss_image_lora.safetensors").exists()
