"""The LoRA trainer's optimizer choices, warmup and progress lines."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from .runs import prepared_lora, progress_lines, run_lora
from .world import Workspace


def _built(ws: Workspace, **train: Any) -> Any:
    ws.add_images("p", 2)
    return prepared_lora(ws.lora_config(train=train))


def test_automagic_groups_the_adapters_by_family_and_records_its_resolved_settings(workspace: Workspace) -> None:
    trainer = _built(workspace, automagic_beta2=0.99, automagic_polarity_history=4, weight_decay=0.01)

    resolved = trainer.run_metadata["optimizer_resolved"]
    assert resolved["groups"] == ["attn", "mlp"] and resolved["fused"] is False
    assert (resolved["beta2"], resolved["polarity_history"], resolved["weight_decay"]) == (0.99, 4, 0.01)
    assert [len(group["params"]) for group in trainer.optimizer.param_groups] == [16, 8]


def test_automagic_param_groups_false_keeps_one_group(workspace: Workspace) -> None:
    trainer = _built(workspace, automagic_param_groups=False)

    assert [group["name"] for group in trainer.optimizer.param_groups] == ["lora"]


def test_a_fused_automagic_trains_the_adapters_too(workspace: Workspace) -> None:
    workspace.add_images("p", 2)

    trainer = run_lora(workspace.lora_config(train={"automagic_fused": True}))

    assert trainer.optimizer.optimizer.fused
    ups = [p for name, p in trainer.lora_network.lora_layers.named_parameters() if name.endswith("lora_up.weight")]
    assert ups and all(torch.count_nonzero(p) > 0 for p in ups)


def test_prodigy_gets_only_the_settings_its_signature_accepts_and_no_none_values(workspace: Workspace) -> None:
    trainer = _built(workspace, optimizer="prodigy_adam", prodigy_d0=1e-5, adam_beta1=0.8, prodigy_beta3=None)

    resolved = trainer.run_metadata["optimizer_resolved"]
    assert resolved["name"] == "prodigy" and resolved["lr"] == 1.0 and resolved["betas"] == (0.8, 0.999)
    assert resolved["d0"] == 1e-5 and "beta3" not in resolved


def test_prodigys_rate_follows_the_learning_rate_once_that_is_a_prodigy_scale(workspace: Workspace) -> None:
    trainer = _built(workspace, optimizer="prodigy", learning_rate=0.5)

    assert trainer.run_metadata["optimizer_resolved"]["lr"] == 0.5


@pytest.mark.parametrize(
    ("name", "recorded"), [("adamw8bit", "adamw8bit"), ("adam8bit", "adam8bit"), ("adamw", "adamw")]
)
def test_the_adam_family_records_its_name_and_rate(workspace: Workspace, name: str, recorded: str) -> None:
    trainer = _built(workspace, optimizer=name, weight_decay=0.1)

    assert trainer.run_metadata["optimizer_resolved"] == {"name": recorded, "lr": 1e-3}
    assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.1


def test_warmup_starts_the_scheduled_rate_low_and_reaches_the_learning_rate(workspace: Workspace) -> None:
    workspace.add_images("p", 2)
    config = workspace.lora_config(train={"optimizer": "adamw", "learning_rate_warmup_steps": 3, "max_steps": 4})
    trainer = prepared_lora(config)

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 / 3)

    trainer.train()

    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert trainer.lr_scheduler.last_epoch == 4


@pytest.mark.parametrize(
    ("train", "tail"),
    [
        ({"optimizer": "automagic3"}, "LR: 1.00e-03 (automagic3: attn 1.00e-03, mlp 1.00e-03)"),
        ({"optimizer": "prodigy"}, "LR: 1.00e-06 (mult 1.000 x d 1.00e-06)"),
        ({"optimizer": "adamw"}, "LR: 1.00e-03"),
    ],
)
def test_each_optimizer_reports_its_rate_in_its_own_way(workspace: Workspace, train: dict[str, Any], tail: str) -> None:
    workspace.add_images("p", 2)

    run_lora(workspace.lora_config(train={**train, "max_steps": 1}))

    assert progress_lines(workspace)[0].endswith(tail)
