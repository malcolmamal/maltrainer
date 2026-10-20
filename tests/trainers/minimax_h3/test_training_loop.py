"""The H3 training loop: epochs, repeats, accumulation, clipping, progress lines and periodic saves."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

import train_minimax_h3_lora as h3
from maltrainer.trainers.minimax_h3 import sample as sample_module

from .runs import prepared, progress_lines, run_h3
from .world import Workspace


def _noting(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make every training step note the file of the sample it trained on."""
    seen: list[str] = []
    original = h3.MiniMaxH3LoRATrainer._training_step

    def noting(self: h3.MiniMaxH3LoRATrainer, sample: sample_module.H3Sample, flow_shift: float) -> Any:
        seen.append(Path(sample.key).name)
        return original(self, sample, flow_shift)

    monkeypatch.setattr(h3.MiniMaxH3LoRATrainer, "_training_step", noting)
    return seen


def _step_fields(line: str) -> tuple[str, ...]:
    match = re.search(r"Step: (\S+) \[(\d+)%\].*Epoch: ([\d/]+)", line)
    assert match is not None
    return match.groups()


def test_every_sample_is_trained_once_per_epoch_and_every_step_is_logged(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(3)
    seen = _noting(monkeypatch)

    trainer = run_h3(workspace.config(train={"num_epochs": 2}))

    assert sorted(seen[:3]) == sorted(seen[3:]) == ["img0.png", "img1.png", "img2.png"]
    lines = progress_lines(workspace)
    assert [_step_fields(line) for line in lines] == [
        ("1/6", "16", "1/2"),
        ("2/6", "33", "1/2"),
        ("3/6", "50", "1/2"),
        ("4/6", "66", "2/2"),
        ("5/6", "83", "2/2"),
        ("6/6", "100", "2/2"),
    ]
    assert (trainer.run_metadata["steps_per_epoch"], trainer.run_metadata["planned_total_steps"]) == (3, 6)
    assert trainer.run_metadata["dataset_samples"] == 3


def test_gradient_accumulation_takes_one_optimizer_step_for_several_samples(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(4)
    seen = _noting(monkeypatch)

    trainer = run_h3(workspace.config(train={"gradient_accumulation_steps": 2}))

    assert len(seen) == 4 and len(progress_lines(workspace)) == 2
    assert trainer.run_metadata["steps_per_epoch"] == 2 and "Step: 2/2 [100%]" in progress_lines(workspace)[-1]


def test_gradients_are_clipped_to_max_grad_norm_before_each_step(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    trainer = prepared(workspace.config(train={"max_grad_norm": 0.25}))
    norms: list[float] = []
    clip = trainer.accelerator.clip_grad_norm_

    def noting(params: Any, max_norm: float) -> Any:
        norms.append(max_norm)
        return clip(params, max_norm)

    monkeypatch.setattr(trainer.accelerator, "clip_grad_norm_", noting)

    trainer.train()

    assert norms == [0.25, 0.25]


def test_checkpoints_are_written_every_save_every_steps_and_once_more_at_the_end(workspace: Workspace) -> None:
    workspace.add_images(5)

    trainer = run_h3(workspace.config(train={"save_every": 2}))

    names = sorted(p.name for p in workspace.out.glob("*.safetensors"))
    assert names == [f"minimax_h3_lora_step_{step}.safetensors" for step in (2, 4, "final")]
    assert trainer.last_saved_path is not None and trainer.last_saved_path.endswith("step_final.safetensors")


def test_the_first_step_logs_the_resolved_sigma_shift_and_latent_size_once(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)

    with caplog.at_level("INFO", logger="maltrainer.trainers.minimax_h3.loop"):
        trainer = run_h3(workspace.config())

    assert trainer.run_metadata["latent_size"] == "2x2" and trainer.run_metadata["resolved_sigma_shift"] == 1.604837
    assert caplog.text.count("Timestep schedule:") == 1
    assert "krea2_shift, sigma shift 1.605 (latent 2x2); audio sigma follows" in caplog.text


@pytest.mark.parametrize(
    ("train", "tail"),
    [
        ({"optimizer": "automagic3"}, r"LR: 1\.00e-03 \(automagic3\), video: [\d.]+ \[image\]$"),
        ({"optimizer": "prodigy"}, r"LR: 1\.00e-06 \(mult 1\.000 x d 1\.00e-06\), video: [\d.]+ \[image\]$"),
        ({"optimizer": "adamw"}, r"LR: 1\.00e-03, video: [\d.]+ \[image\]$"),
    ],
)
def test_each_optimizer_reports_its_rate_in_its_own_way(workspace: Workspace, train: dict[str, Any], tail: str) -> None:
    workspace.add_images(1)

    run_h3(workspace.config(train={**train, "automagic_min_lr": 1e-3, "automagic_max_lr": 1e-3}))

    assert re.search(tail, progress_lines(workspace)[0])


def test_each_line_names_the_kind_and_the_loss_parts_of_its_sample(workspace: Workspace) -> None:
    workspace.add_images(1)
    workspace.add_clip()
    workspace.add_voice()

    run_h3(workspace.config())

    details = sorted(
        re.sub(r"[\d.]+", "#", line.split("LR: ")[1].split(", ", 1)[1]) for line in progress_lines(workspace)
    )
    assert details == ["audio: # [audio]", "video: # [image]", "video: #, audio: # [clip]"]


def test_a_second_process_saves_nothing(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared(workspace.config())
    trainer.is_main_process = False

    trainer.save_checkpoint("final")

    assert not list(workspace.out.glob("*.safetensors")) and trainer.last_saved_path is None


def test_the_warmup_scheduler_advances_with_every_optimizer_step(workspace: Workspace) -> None:
    workspace.add_images(4)

    trainer = run_h3(workspace.config(train={"learning_rate_warmup_steps": 3}))

    assert trainer.lr_scheduler.last_epoch == 4
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_accumulation_over_a_sample_count_it_does_not_divide_plans_the_steps_it_takes(workspace: Workspace) -> None:
    workspace.add_images(3)

    trainer = run_h3(workspace.config(train={"num_epochs": 2, "gradient_accumulation_steps": 2}))

    lines = progress_lines(workspace)
    assert [_step_fields(line)[:2] for line in lines] == [("1/3", "33"), ("2/3", "66"), ("3/3", "100")]
    assert trainer.run_metadata["planned_total_steps"] == 3
