"""The finetune loop: epochs and steps, saves on schedule, the health monitor and the dataset fingerprint."""

from __future__ import annotations

import logging
import re
import shutil

import pytest
import torch

from train_krea2_finetune import Krea2FinetuneTrainer
from train_krea2_lora import Krea2LoRATrainer

from .runs import progress_lines, run_finetune
from .world import Workspace, write_image


def test_max_steps_ends_the_run_in_the_middle_of_an_epoch_and_the_counters_say_where(workspace: Workspace) -> None:
    workspace.add_images(3)

    trainer = run_finetune(workspace.finetune_config(train={"max_steps": 7}))

    lines = progress_lines(workspace)
    assert len(lines) == 7 and re.search(
        r"Step: 7/7 \[100%\], Loss: \d\.\d{6}, Epoch: 3/3, Window: mlp.down, LR: ", lines[-1]
    )
    assert (trainer.global_step, trainer.epoch, trainer.position) == (7, 2, 1)
    assert (trainer.run_metadata["steps_per_epoch"], trainer.run_metadata["planned_total_steps"]) == (3, 7)


def test_without_max_steps_the_epochs_decide(workspace: Workspace) -> None:
    workspace.add_images(2)

    trainer = run_finetune(workspace.finetune_config(train={"max_steps": 0, "num_epochs": 2}))

    assert len(progress_lines(workspace)) == 4 and (trainer.epoch, trainer.position) == (1, 2)


def test_repeats_multiply_the_steps_of_an_epoch(workspace: Workspace) -> None:
    workspace.add_images(2)

    trainer = run_finetune(workspace.finetune_config(dataset={"repeats": 3}, train={"max_steps": 0, "num_epochs": 1}))

    assert trainer.global_step == 6


def test_a_dataset_whose_every_file_fails_to_load_is_refused_when_training_starts(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.data.mkdir()
    (workspace.data / "broken.png").write_bytes(b"not an image")

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(ValueError, match=r"No usable images: every file in the dataset failed"),
    ):
        run_finetune(workspace.finetune_config())

    assert "Skipping broken.png:" in caplog.text


def test_save_every_writes_numbered_models_with_a_state_but_never_for_the_final_step(workspace: Workspace) -> None:
    workspace.add_images(2)

    run_finetune(workspace.finetune_config(train={"max_steps": 3, "save_every": 1}))

    assert sorted(path.name for path in workspace.out.glob("*.safetensors")) == [
        "ft.safetensors",
        "ft_step_1.safetensors",
        "ft_step_2.safetensors",
    ]


def test_the_monitor_summarises_every_monitor_every_steps_and_at_the_last_one(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)

    with caplog.at_level(logging.INFO):
        run_finetune(workspace.finetune_config(train={"max_steps": 3, "monitor_every": 2}))

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[monitor]")]
    assert [line.split(" | ")[0] for line in lines] == ["[monitor] step 2/3", "[monitor] step 3/3"]
    assert "s/step" in lines[0] and "nonfinite 0" in lines[0]


def test_three_nan_losses_in_a_row_stop_the_run_without_saving_anything(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)

    def nan_step(
        _self: Krea2LoRATrainer, _index: int, _shift_logged: list[bool]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = torch.ones((), requires_grad=True)
        return weight * float("nan"), torch.zeros(1), torch.tensor(0.5)

    monkeypatch.setattr(Krea2LoRATrainer, "_training_step", nan_step)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match=r"the loss was NaN / inf 3 steps in a row"):
        run_finetune(workspace.finetune_config(train={"max_steps": 6}))

    assert not list(workspace.out.glob("*.safetensors")) and "An error occurred during training" in caplog.text


def _fingerprint(ws: Workspace, **dataset: int) -> str:
    trainer = Krea2FinetuneTrainer(ws.finetune_config(dataset=dataset))
    trainer.precache_dataset()
    return trainer._dataset_fingerprint()


def test_the_fingerprint_follows_the_captions_the_repeats_and_the_relative_paths_not_the_folder(
    workspace: Workspace,
) -> None:
    workspace.add_images(2)
    original = _fingerprint(workspace)

    moved = workspace.root / "moved"
    shutil.copytree(workspace.data, moved / "data")
    shutil.copyfile(workspace.checkpoint, Workspace(moved).checkpoint)  # the base model is checked before caching
    assert _fingerprint(Workspace(moved)) == original
    assert _fingerprint(workspace, repeats=2) != original
    write_image(workspace.data / "img1.png", "another caption")
    assert _fingerprint(workspace) != original
    assert len(original) == 40
