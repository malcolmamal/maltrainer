"""Continuing from a state: what it checks, what changes when the data or the settings changed."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

from .runs import cached_finetune, run_finetune
from .world import Workspace


def _first_run(ws: Workspace, images: int = 3, **sections: Any) -> None:
    ws.add_images("p", images)
    train = {"max_steps": 2, **sections.pop("train", {})}
    run_finetune(ws.config(train=train, **sections))


def _resume(ws: Workspace, steps: int = 4, **sections: Any) -> Any:
    train = {"max_steps": steps, "resume_from": str(ws.out / "ft_state"), **sections.pop("train", {})}
    return run_finetune(ws.config(train=train, **sections))


@pytest.mark.parametrize("missing", ["model.safetensors", "optimizer.pt", "trainer_state.pt"])
def test_a_state_folder_with_a_missing_file_is_refused(workspace: Workspace, missing: str) -> None:
    _first_run(workspace)
    (workspace.out / "ft_state" / missing).unlink()

    with pytest.raises(FileNotFoundError, match=rf"train\.resume_from: .*{missing} is missing"):
        _resume(workspace)


def test_a_state_saved_with_other_optimizer_groups_is_refused(workspace: Workspace) -> None:
    _first_run(workspace)

    with pytest.raises(
        ValueError, match=r"The resumed optimizer had groups \['attn', 'mlp'\], this config builds \['all'\]"
    ):
        _resume(workspace, train={"automagic_param_groups": False})


def test_a_state_saved_with_a_different_number_of_trained_parameters_is_refused(workspace: Workspace) -> None:
    _first_run(workspace)

    with pytest.raises(ValueError, match=r"doesn't match the size of optimizer's group"):
        _resume(workspace, finetune={"trainable_prefixes": ["layers.1."]})


def test_a_resume_that_is_already_at_its_step_budget_trains_nothing_and_says_so(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    _first_run(workspace)
    before = sorted(path.name for path in workspace.out.iterdir())

    with caplog.at_level(logging.WARNING):
        trainer = _resume(workspace, steps=2)

    assert "already at step 2 of 2; raise train.max_steps to continue" in caplog.text
    assert trainer.global_step == 2
    assert sorted(path.name for path in workspace.out.iterdir()) == before


def test_a_resume_logs_where_in_its_epoch_it_continues(workspace: Workspace, caplog: pytest.LogCaptureFixture) -> None:
    _first_run(workspace)

    with caplog.at_level(logging.INFO):
        trainer = _resume(workspace, steps=3)

    assert "Resumed at step 2 (epoch 1, sample 2 of its order)" in caplog.text
    assert (trainer.global_step, trainer.epoch, trainer.position) == (3, 0, 3)


def test_a_changed_dataset_continues_with_a_fresh_epoch_and_a_warning(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    _first_run(workspace)
    workspace.add_images("new", 1)

    with caplog.at_level(logging.WARNING):
        trainer = _resume(workspace, steps=5)

    assert (
        "The dataset changed since the saved state (3 samples then, 4 now); resuming at step 2 with a fresh epoch."
        in (caplog.text)
    )
    assert (trainer.global_step, trainer.epoch, trainer.position) == (5, 1, 3)


def test_a_dataset_moved_to_another_folder_still_counts_as_the_same(workspace: Workspace) -> None:
    _first_run(workspace)
    shutil.move(workspace.data, workspace.root / "moved")
    config = workspace.config(dataset={"path": str(workspace.root / "moved")})
    config["train"].update(max_steps=3, resume_from=str(workspace.out / "ft_state"))

    trainer = run_finetune(config)

    assert (trainer.global_step, trainer.epoch, trainer.position) == (3, 0, 3)


def test_the_fingerprint_follows_the_samples_their_captions_and_the_repeats(workspace: Workspace) -> None:
    workspace.add_images("p", 2)
    workspace.add_images(workspace.root / "reg", 1, caption="a person")

    def fingerprint(**dataset: Any) -> str:
        return cached_finetune(workspace.config(dataset=dataset))._dataset_fingerprint()

    base = fingerprint()
    (workspace.data / "p" / "img0.txt").write_text("a different caption", encoding="utf-8")

    assert fingerprint() != base
    assert fingerprint(repeats=2) != fingerprint()
    assert fingerprint(folder_repeats={"p": 2}) not in (fingerprint(), fingerprint(repeats=2))
    assert fingerprint(reg_path=str(workspace.root / "reg")) != fingerprint()


def test_resuming_an_ema_run_without_ema_ignores_the_saved_average(workspace: Workspace) -> None:
    _first_run(workspace, train={"ema_decay": 0.5})

    trainer = _resume(workspace)

    assert (trainer.ema, trainer.global_step) == (None, 4)


def test_turning_ema_on_when_resuming_a_run_that_had_none_starts_the_average_from_the_current_weights(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    _first_run(workspace)

    with caplog.at_level(logging.WARNING):
        trainer = _resume(workspace, train={"ema_decay": 0.5})

    assert "The resumed run had no EMA; the average starts from the current weights." in caplog.text
    assert trainer.ema is not None and trainer.ema.updates == 2


def test_an_ema_state_that_lacks_the_trained_parameters_of_this_config_is_refused(workspace: Workspace) -> None:
    _first_run(workspace, train={"ema_decay": 0.5}, finetune={"trainable_prefixes": ["layers.0."]})

    with pytest.raises(
        ValueError, match=r"EMA state lacks 6 trained parameters, e\.g\. \['layers\.1\.proj\.q\.weight'"
    ):
        _resume(workspace, train={"ema_decay": 0.5}, finetune={"trainable_prefixes": ["layers.1."]})


def test_the_state_folder_can_be_given_as_a_path_object_of_any_spelling(workspace: Workspace) -> None:
    _first_run(workspace)
    config = workspace.config(train={"max_steps": 3, "resume_from": Path(workspace.out / "ft_state").as_posix()})

    assert run_finetune(config).global_step == 3


def test_a_state_saved_before_any_step_of_an_epoch_keeps_that_epoch_when_the_dataset_changed(
    workspace: Workspace,
) -> None:
    _first_run(workspace)
    state = workspace.out / "ft_state" / "trainer_state.pt"
    counters = torch.load(state, map_location="cpu", weights_only=False)
    torch.save({**counters, "position": 0}, state)
    workspace.add_images("new", 1)

    trainer = _resume(workspace, steps=4)

    assert (trainer.global_step, trainer.epoch, trainer.position) == (4, 0, 2)
