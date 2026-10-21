"""How a config becomes windows: rotation order, custom windows, always-on text fusion, refused setups."""

from __future__ import annotations

import logging
from typing import Any

import pytest
import torch
from accelerate import Accelerator

from train_krea2_finetune import Krea2FinetuneTrainer

from .runs import final_weights, prepared_finetune, progress_lines, run_finetune
from .world import Workspace, stored_weights


def _loaded(ws: Workspace, **sections: Any) -> Krea2FinetuneTrainer:
    ws.add_images(1)
    return prepared_finetune(ws.finetune_config(**sections))


@pytest.mark.parametrize(("step", "window"), [(0, 0), (1, 0), (2, 1), (5, 2), (7, 3), (8, 0), (17, 0), (19, 1)])
def test_the_window_follows_the_step_through_the_cycle_of_four_every_rotate_every_steps(
    workspace: Workspace, step: int, window: int
) -> None:
    trainer = _loaded(workspace)

    assert trainer._window_at(step) == window


@pytest.mark.parametrize(
    ("finetune", "every"), [({}, 100), ({"rotate_every": None}, 100), ({"rotate_every": 7}, 7), (None, 100)]
)
def test_rotate_every_defaults_to_one_hundred_steps(workspace: Workspace, finetune: Any, every: int) -> None:
    workspace.add_images(1)
    config = workspace.finetune_config()
    config["finetune"] = finetune

    assert Krea2FinetuneTrainer(config)._rotate_every() == every


@pytest.mark.parametrize("value", [0, -3])
def test_a_rotate_every_below_one_is_refused_at_construction_not_read_as_the_default(
    workspace: Workspace, value: int
) -> None:
    workspace.add_images(1)
    config = workspace.finetune_config(finetune={"rotate_every": value})

    with pytest.raises(ValueError, match=r"finetune\.rotate_every must be at least 1"):
        Krea2FinetuneTrainer(config)


def test_each_window_comes_up_in_turn_and_the_log_says_which(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(3)

    with caplog.at_level(logging.INFO):
        run_finetune(workspace.finetune_config(finetune={"rotate_every": 1}, train={"max_steps": 5}))

    swaps = [r.getMessage().split(" (")[0] for r in caplog.records if "training window" in r.getMessage()]
    assert swaps == [
        f"Step {i}: training window {n}"
        for i, n in enumerate(["attention", "mlp.gate", "mlp.up", "mlp.down", "attention"])
    ]
    assert [line.split("Window: ")[1].split(",")[0] for line in progress_lines(workspace)] == [
        "attention",
        "mlp.gate",
        "mlp.up",
        "mlp.down",
        "attention",
    ]


def test_custom_windows_replace_the_default_four_and_leave_other_linears_frozen(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    windows = [{"name": "queries", "prefixes": ["attn.wq"]}]

    with caplog.at_level(logging.WARNING):
        trainer = run_finetune(workspace.finetune_config(finetune={"windows": windows}, train={"max_steps": 3}))

    before, after = stored_weights(workspace.checkpoint), final_weights(workspace.out)
    changed = {k for k in before if not torch.equal(before[k], after["model.diffusion_model." + k])}
    assert {k for k in changed if k.startswith("blocks.")} == {"blocks.0.attn.wq.weight", "blocks.1.attn.wq.weight"}
    assert trainer.rotator.names == ["queries"]
    assert "14 block Linears are in no finetune window and stay frozen, e.g." in caplog.text


@pytest.mark.parametrize(
    ("windows", "message"),
    [
        ([{"name": "nothing", "prefixes": ["nope."]}], r"finetune window 'nothing' \(\['nope\.'\]\) matches no Linear"),
        (
            [{"name": "a", "prefixes": ["attn."]}, {"name": "b", "prefixes": ["attn.wq"]}],
            r"blocks\.0\.attn\.wq\.weight is in two finetune windows, 'a' and 'b'",
        ),
    ],
)
def test_windows_that_match_nothing_or_overlap_are_refused(
    workspace: Workspace, windows: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _loaded(workspace, finetune={"windows": windows})


def test_the_text_fusion_trains_all_run_with_its_own_optimizer_and_always_on_can_switch_it_off(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        with_fusion = _loaded(workspace)
        without = _loaded(workspace, finetune={"always_on": []})

    assert with_fusion.always_optimizer is not None and len(with_fusion.always_on) == 33
    assert with_fusion.run_metadata["always_on"] == ["txtfusion"]
    assert with_fusion.run_metadata["always_on_parameters"] == 69635
    assert without.always_optimizer is None and without.run_metadata["always_on"] == []
    assert "always trained: txtfusion (0.00B)" in caplog.text and "always trained: nothing (0.00B)" in caplog.text


def test_the_optimizer_is_fused_automagic3_pinned_to_the_learning_rate_by_default(workspace: Workspace) -> None:
    workspace.add_images(1)
    config = workspace.finetune_config()
    for key in ("learning_rate", "automagic_min_lr", "automagic_max_lr", "automagic_polarity_history"):
        del config["train"][key]

    trainer = prepared_finetune(config)

    assert trainer.run_metadata["optimizer_resolved"] == {
        "name": "automagic3", "lr": 1e-5, "min_lr": 1e-5, "max_lr": 1e-5, "beta2": 0.999, "eps": 1e-30,
        "clip_threshold": 1.0, "weight_decay": 0.0, "polarity_history": 8, "fused": True,
    }  # fmt: skip


@pytest.mark.parametrize("name", ["automagic3", "automagic", "automagic_v3", "AUTOMAGIC"])
def test_every_automagic_spelling_is_accepted(workspace: Workspace, name: str) -> None:
    assert _loaded(workspace, train={"optimizer": name}).rotator is not None


def test_any_other_optimizer_is_refused(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match=r"Krea 2 finetuning supports automagic3 only \(got 'adamw'\)"):
        _loaded(workspace, train={"optimizer": "adamw"})


@pytest.mark.parametrize(
    ("sections", "message"),
    [
        ({"train": {"dop": True}}, r"train\.dop is the LoRA trainer's; the Krea 2 finetune has no prior preservation"),
        ({"train": {"gradient_accumulation_steps": 2}}, r"gradient_accumulation_steps must be 1"),
        ({"train": {"batch_size": 2}}, r"Krea 2 trains at batch_size 1"),
    ],
)
def test_setups_the_finetune_cannot_run_are_refused_up_front(
    workspace: Workspace, sections: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Krea2FinetuneTrainer(workspace.finetune_config(**sections))


def test_a_finetune_runs_on_one_process_only(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Accelerator, "num_processes", property(lambda _self: 2))

    with pytest.raises(ValueError, match=r"Krea 2 finetuning runs on one GPU"):
        Krea2FinetuneTrainer(workspace.finetune_config())
