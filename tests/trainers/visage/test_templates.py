"""The two shipped templates run through the trainer, and each trains with the optimizer it names."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from maltrainer.utils.lora_metadata import read_metadata

from .runs import resolved_optimizer, run_lora
from .world import Workspace

TEMPLATES = Path(__file__).resolve().parents[3] / "templates"


def sections(name: str) -> dict[str, Any]:
    """A template's settings without the paths the workspace points at its own files."""
    text = (TEMPLATES / name).read_text(encoding="utf-8").replace("[person_to_train]", "zed")
    config: dict[str, Any] = yaml.safe_load(text)
    del config["model"], config["dataset"]["path"], config["output"]["path"]
    return config


def run_template(ws: Workspace, name: str) -> Any:
    """The template on the tiny world: 32 px, fp32, 3 optimizer steps of one sample."""
    ws.add_images(3)
    config = ws.config(
        sections(name),
        dataset={"resolution": 32},
        train={"max_steps": 3, "gradient_accumulation_steps": 1},
        dtype="float32",
        mixed_precision="no",
        lora={"rank": 4, "alpha": 4},
        output={"dtype": "bfloat16"},
    )
    return run_lora(config)


@pytest.mark.parametrize(
    ("template", "optimizer", "rate", "saved"),
    [
        ("visage_template.yaml", "automagic3", 1e-6, "visage_zed.safetensors"),
        ("visage_adamw_template.yaml", "adamw", 1e-4, "visage_adamw_zed.safetensors"),
    ],
)
def test_each_template_trains_with_its_optimizer_and_saves_its_lora(
    workspace: Workspace, template: str, optimizer: str, rate: float, saved: str
) -> None:
    trainer = run_template(workspace, template)

    resolved = resolved_optimizer(trainer)
    assert (resolved["name"], resolved["lr"]) == (optimizer, rate)
    assert (workspace.out / saved).exists()


def test_the_automagic_template_adapts_attention_and_mlp_separately_and_keeps_the_ema(workspace: Workspace) -> None:
    trainer = run_template(workspace, "visage_template.yaml")

    assert resolved_optimizer(trainer)["groups"] == ["attn", "mlp"]  # the blocks preset has no "other" Linears
    assert (trainer.run_metadata["resolved_sigma_shift"], trainer.ema is not None) == (1.0, True)


def test_the_adamw_template_is_a_flat_rate_with_decay(workspace: Workspace) -> None:
    trainer = run_template(workspace, "visage_adamw_template.yaml")

    optimizer = trainer.optimizer.optimizer
    assert type(optimizer) is torch.optim.AdamW
    assert (optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["weight_decay"]) == (1e-4, 0.01)
    assert trainer.lr_scheduler is None
    assert read_metadata(workspace.out / "visage_adamw_zed.safetensors")["maltrainer_lora_layers"] == "8"


def test_both_templates_share_the_model_the_recipe_and_the_preview_settings() -> None:
    plain, adamw = sections("visage_template.yaml"), sections("visage_adamw_template.yaml")
    differing = {"optimizer", "learning_rate", "learning_rate_warmup_steps", "weight_decay"} | {
        key for key in plain["train"] if key.startswith("automagic_")
    }

    assert {key: value for key, value in plain["train"].items() if key not in differing} == {
        key: value for key, value in adamw["train"].items() if key not in differing
    }
    assert plain["lora"] == adamw["lora"] and plain["sample"] == adamw["sample"]
    assert plain["dataset"] == adamw["dataset"] and plain["train"]["save_every"] == 0
