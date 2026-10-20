"""setup_optimizer: Automagic3 with module-family groups, Prodigy, AdamW, 8-bit AdamW, warmup, and the progress line."""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from prodigyopt import Prodigy

from maltrainer.optimizers import Automagic3

from .runs import prepared, progress_lines, resolved_optimizer, run_lora
from .world import Workspace


def _resolved(ws: Workspace, **train: Any) -> dict[str, Any]:
    ws.add_images(1)
    return resolved_optimizer(prepared(ws.config(train=train)))


@pytest.mark.parametrize("name", ["automagic3", "automagic", "automagic_v3", "AUTOMAGIC3"])
def test_every_automagic_spelling_builds_automagic3_with_one_group_per_module_family(
    workspace: Workspace, name: str
) -> None:
    workspace.add_images(1)

    trainer = prepared(workspace.config(train={"optimizer": name}))

    assert isinstance(trainer.optimizer, Automagic3)
    assert [group["name"] for group in trainer.optimizer.param_groups] == ["attn", "mlp"]
    assert sum(len(group["params"]) for group in trainer.optimizer.param_groups) == 2 * 24


def test_the_input_and_output_layers_get_a_group_of_their_own_with_the_all_preset(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared(workspace.config(lora={"targets": "all"}))

    assert [group["name"] for group in trainer.optimizer.param_groups] == ["other", "attn", "mlp"]


def test_the_resolved_automagic_settings_are_the_config_over_the_defaults(workspace: Workspace) -> None:
    resolved = _resolved(workspace, automagic_fused=True, automagic_beta2=0.99, weight_decay=0.01)

    assert resolved == {
        "name": "automagic3",
        "groups": ["attn", "mlp"],
        "lr": 1e-3,
        "min_lr": 1e-3,
        "max_lr": 1e-3,
        "beta2": 0.99,
        "eps": 1e-30,
        "clip_threshold": 1.0,
        "weight_decay": 0.01,
        "polarity_history": 2,
        "fused": True,
    }


def test_the_defaults_are_a_1e_6_launch_point_inside_a_wide_rate_range(workspace: Workspace) -> None:
    workspace.add_images(1)
    config = workspace.config()
    for key in ("learning_rate", "automagic_min_lr", "automagic_max_lr", "automagic_polarity_history"):
        del config["train"][key]

    resolved = resolved_optimizer(prepared(config))

    assert (resolved["lr"], resolved["min_lr"], resolved["max_lr"], resolved["polarity_history"]) == (
        1e-6,
        1e-8,
        1e3,
        16,
    )


def test_without_parameter_groups_every_adapter_shares_one_rate(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared(workspace.config(train={"automagic_param_groups": False}))

    assert [group["name"] for group in trainer.optimizer.param_groups] == ["lora"]


def test_prodigy_keeps_the_settings_this_install_accepts_and_drops_an_unset_beta3(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared(workspace.config(train={"optimizer": "prodigy", "prodigy_d0": 1e-5, "adam_beta1": 0.8}))

    assert isinstance(trainer.optimizer, Prodigy)
    resolved = resolved_optimizer(trainer)
    assert (resolved["name"], resolved["lr"], resolved["d0"], resolved["betas"]) == ("prodigy", 1.0, 1e-5, (0.8, 0.999))
    assert "beta3" not in resolved and resolved["decouple"] is True


def test_a_learning_rate_of_one_hundredth_or_more_is_prodigys_multiplier(workspace: Workspace) -> None:
    assert _resolved(workspace, optimizer="prodigy", learning_rate=0.5)["lr"] == 0.5
    assert _resolved(workspace, optimizer="prodigy", learning_rate=0.5, prodigy_lr=2.0)["lr"] == 2.0


def test_any_other_name_is_plain_adamw(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared(workspace.config(train={"optimizer": "adamw", "weight_decay": 0.1}))

    assert type(trainer.optimizer) is torch.optim.AdamW
    assert trainer.run_metadata["optimizer_resolved"] == {"name": "adamw", "lr": 1e-3}
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.1


@pytest.mark.parametrize("name", ["adamw8bit", "adam8bit"])
def test_the_8_bit_optimizers_come_from_bitsandbytes(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    class FakeAdamW8bit(torch.optim.AdamW):
        pass

    optim = SimpleNamespace(AdamW8bit=FakeAdamW8bit)
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace(optim=optim))
    workspace.add_images(1)

    trainer = prepared(workspace.config(train={"optimizer": name}))

    assert isinstance(trainer.optimizer, FakeAdamW8bit)
    assert trainer.run_metadata["optimizer_resolved"] == {"name": name, "lr": 1e-3}


def test_warmup_scales_the_rate_up_over_the_first_steps_and_without_it_there_is_no_scheduler(
    workspace: Workspace,
) -> None:
    workspace.add_images(2)

    plain = prepared(workspace.config(train={"optimizer": "adamw"}))
    warmed = run_lora(workspace.config(train={"optimizer": "adamw", "learning_rate_warmup_steps": 10}))

    assert plain.lr_scheduler is None
    assert warmed.lr_scheduler is not None
    assert warmed.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 * 4 / 10)


@pytest.mark.parametrize(
    ("train", "expected"),
    [
        ({}, r"LR: 1\.00e-03 \(automagic3: attn 1\.00e-03, mlp 1\.00e-03\)$"),
        ({"optimizer": "prodigy"}, r"LR: 1\.00e-06 \(mult 1\.000 x d 1\.00e-06\)$"),
        ({"optimizer": "adamw"}, r"LR: 1\.00e-03$"),
    ],
)
def test_the_progress_line_shows_the_rate_the_optimizer_actually_uses(
    workspace: Workspace, train: dict[str, Any], expected: str
) -> None:
    workspace.add_images(1)

    run_lora(workspace.config(train={"max_steps": 1, **train}))

    assert re.search(r"^.*Step: 1/1 \[100%\], Loss: \d\.\d{6}, Epoch: 1/1, " + expected, progress_lines(workspace)[0])
