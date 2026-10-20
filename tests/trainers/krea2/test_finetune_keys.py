"""Key commands while a finetune trains: save (with its state), quit, pause and help."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest
from accelerate import Accelerator

import train_krea2_finetune as finetune
import train_krea2_lora as lora

from .fakes import ScriptedKeys, script_keys
from .runs import final_weights, run_finetune
from .world import Workspace, same_values


@pytest.fixture
def press(monkeypatch: pytest.MonkeyPatch) -> Callable[[list[list[str]]], list[ScriptedKeys]]:
    """``press([[], ['s']])``: the keys pending at the 1st and the 2nd step boundary."""

    def pressing(script: list[list[str]]) -> list[ScriptedKeys]:
        return script_keys(monkeypatch, [finetune, lora], script)

    return pressing


def _saved(ws: Workspace) -> list[str]:
    return sorted(path.name for path in ws.out.glob("*.safetensors"))


def test_s_saves_a_numbered_checkpoint_with_its_state_and_training_continues(
    workspace: Workspace, press: Callable[..., Any]
) -> None:
    workspace.add_images(2)
    press([[], ["s"]])

    trainer = run_finetune(workspace.finetune_config())

    assert _saved(workspace) == ["ft.safetensors", "ft_step_2.safetensors"]
    assert trainer.global_step == 3 and (workspace.out / "ft_state" / "trainer_state.pt").exists()


def test_q_saves_and_stops_without_a_final_file_leaving_a_state_to_resume_from(
    workspace: Workspace, press: Callable[..., Any], caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    press([[], ["q"]])

    with caplog.at_level(logging.INFO):
        trainer = run_finetune(workspace.finetune_config(train={"max_steps": 5}))

    assert trainer.global_step == 2 and _saved(workspace) == ["ft_step_2.safetensors"]
    assert (workspace.out / "ft_state" / "trainer_state.pt").exists()
    assert "Krea 2 finetuning stopped by key at step 2; resume from the _state folder." in caplog.text


def test_a_run_stopped_with_q_resumes_to_the_same_model_as_a_straight_run(
    workspace: Workspace, press: Callable[..., Any]
) -> None:
    workspace.add_images(3)
    straight = workspace.finetune_config(train={"max_steps": 5}, output={"path": str(workspace.root / "straight")})
    run_finetune(straight)
    press([[], [], ["q"]])
    run_finetune(workspace.finetune_config(train={"max_steps": 5}))
    press([])

    state = str(workspace.out / "ft_state")
    run_finetune(workspace.finetune_config(train={"max_steps": 5, "resume_from": state}))

    assert same_values(final_weights(workspace.out), final_weights(workspace.root / "straight"))


def test_q_does_not_stop_on_a_save_that_failed(
    workspace: Workspace, press: Callable[..., Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    real, calls = finetune.save_krea2_model, []

    def flaky(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise OSError("disk full")
        real(*args, **kwargs)

    monkeypatch.setattr(finetune, "save_krea2_model", flaky)
    press([["q"]])

    with caplog.at_level(logging.ERROR):
        trainer = run_finetune(workspace.finetune_config())

    assert trainer.global_step == 3 and _saved(workspace) == ["ft.safetensors"]
    assert "Key [q]: step 1 was not saved, so training continues" in caplog.text


def test_p_pauses_until_p_again_and_h_lists_the_keys(
    workspace: Workspace, press: Callable[..., Any], caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    press([["p"], ["h", "p"]])

    with caplog.at_level(logging.INFO):
        trainer = run_finetune(workspace.finetune_config(train={"max_steps": 2}))

    assert trainer.global_step == 2
    assert "Paused at step 1: [p] resume" in caplog.text and "Resumed at step 1 after" in caplog.text
    assert "[s] save a checkpoint now" in caplog.text


def test_a_process_that_is_not_the_main_one_listens_for_no_keys(
    workspace: Workspace, press: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Accelerator, "is_main_process", property(lambda _self: False))
    workspace.add_images(2)
    made = press([["q"]])

    trainer = run_finetune(workspace.finetune_config(train={"max_steps": 2}))

    assert made == [] and trainer.global_step == 2
