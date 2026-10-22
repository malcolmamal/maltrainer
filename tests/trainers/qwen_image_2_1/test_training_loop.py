"""The Qwen training loop: epochs, max_steps, accumulation, clipping, progress lines and periodic saves."""

from __future__ import annotations

import re
from typing import Any

import pytest

from .runs import prepared, progress_lines, run_qwen
from .world import Workspace


def test_max_steps_stops_mid_epoch_and_the_log_counts_every_step(workspace: Workspace) -> None:
    workspace.add_images(3)

    trainer = run_qwen(workspace.config(train={"max_steps": 5}))

    assert [line.split("Step: ")[1].split(",")[0] for line in progress_lines(workspace)] == [
        "1/5 [20%]",
        "2/5 [40%]",
        "3/5 [60%]",
        "4/5 [80%]",
        "5/5 [100%]",
    ]
    assert "Epoch: 1/2" in progress_lines(workspace)[0] and "Epoch: 2/2" in progress_lines(workspace)[-1]
    assert (trainer.run_metadata["steps_per_epoch"], trainer.run_metadata["planned_total_steps"]) == (3, 5)
    assert trainer.run_metadata["dataset_samples"] == 3


def test_repeats_make_the_epoch_longer_and_num_epochs_sets_the_total(workspace: Workspace) -> None:
    workspace.add_images(3)

    trainer = run_qwen(workspace.config(dataset={"repeats": 2}, train={"max_steps": 0, "num_epochs": 2}))

    assert (trainer.run_metadata["steps_per_epoch"], trainer.run_metadata["planned_total_steps"]) == (6, 12)
    assert len(progress_lines(workspace)) == 12


def test_gradient_accumulation_takes_one_optimizer_step_for_several_micro_batches(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(4)
    trainer = prepared(workspace.config(train={"gradient_accumulation_steps": 2, "max_steps": 2}))
    forwards: list[int] = []
    forward = trainer.lora_network.forward

    def counting(**kwargs: Any) -> Any:
        forwards.append(1)
        return forward(**kwargs)

    monkeypatch.setattr(trainer.lora_network, "forward", counting)

    trainer.train()

    assert len(forwards) == 4 and len(progress_lines(workspace)) == 2
    assert trainer.run_metadata["steps_per_epoch"] == 2


def test_an_epoch_that_ends_mid_accumulation_still_takes_its_last_step(workspace: Workspace) -> None:
    workspace.add_images(3)

    trainer = run_qwen(workspace.config(train={"gradient_accumulation_steps": 2, "max_steps": 4}))

    assert trainer.run_metadata["steps_per_epoch"] == 2 and len(progress_lines(workspace)) == 4
    assert (workspace.out / "qwen_image_2_1_lora.safetensors").exists()


def test_gradients_are_clipped_to_max_grad_norm_before_each_step(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    trainer = prepared(workspace.config(train={"max_grad_norm": 0.25, "max_steps": 2}))
    norms: list[float] = []
    clip = trainer.accelerator.clip_grad_norm_

    def noting(params: Any, max_norm: float) -> Any:
        norms.append(max_norm)
        return clip(params, max_norm)

    monkeypatch.setattr(trainer.accelerator, "clip_grad_norm_", noting)

    trainer.train()

    assert norms == [0.25, 0.25]


def test_the_clip_norm_defaults_to_one(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace.add_images(1)
    trainer = prepared(workspace.config(train={"max_steps": 1}))
    norms: list[float] = []
    clip = trainer.accelerator.clip_grad_norm_

    def noting(params: Any, max_norm: float) -> Any:
        norms.append(max_norm)
        return clip(params, max_norm)

    monkeypatch.setattr(trainer.accelerator, "clip_grad_norm_", noting)

    trainer.train()

    assert norms == [1.0]


def test_checkpoints_are_written_every_save_every_steps_and_the_last_step_is_only_the_final_one(
    workspace: Workspace,
) -> None:
    workspace.add_images(2)

    trainer = run_qwen(workspace.config(train={"max_steps": 6, "save_every": 2}))

    names = sorted(p.name for p in workspace.out.glob("*.safetensors"))
    assert names == [
        "qwen_image_2_1_lora.safetensors",
        "qwen_image_2_1_lora_step_2.safetensors",
        "qwen_image_2_1_lora_step_4.safetensors",
    ]
    assert trainer.last_saved_path is not None and trainer.last_saved_path.endswith("qwen_image_2_1_lora.safetensors")


@pytest.mark.parametrize(
    ("train", "tail"),
    [
        ({"optimizer": "automagic3"}, r"LR: 1\.00e-03 \(automagic3\)$"),
        ({"optimizer": "prodigy"}, r"LR: 1\.00e-06 \(mult 1\.000 x d 1\.00e-06\)$"),
        ({"optimizer": "adamw"}, r"LR: 1\.00e-03$"),
    ],
)
def test_each_optimizer_reports_its_rate_in_its_own_way(workspace: Workspace, train: dict[str, Any], tail: str) -> None:
    workspace.add_images(1)

    run_qwen(workspace.config(train={**train, "max_steps": 1, "automagic_min_lr": 1e-3, "automagic_max_lr": 1e-3}))

    assert re.search(tail, progress_lines(workspace)[0])


def test_the_warmup_scheduler_advances_with_every_optimizer_step(workspace: Workspace) -> None:
    workspace.add_images(2)

    trainer = run_qwen(workspace.config(train={"warmup_steps": 2, "max_steps": 4}))

    assert trainer.lr_scheduler.last_epoch == 4
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_a_second_process_saves_nothing(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared(workspace.config())
    trainer.is_main_process = False

    trainer.save_checkpoint("final")

    assert not list(workspace.out.glob("*.safetensors")) and trainer.last_saved_path is None
