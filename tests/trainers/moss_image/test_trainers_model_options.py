"""Model-side options both trainers share: gradient checkpointing, torch.compile and a config with no output."""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch
import yaml

import train_moss_image_finetune as finetune
import train_moss_image_lora as lora

from .runs import prepared_finetune, prepared_lora, run_finetune, same_values
from .world import Workspace, named_weights

COMPILE = {"compile": True, "compile_mode": "reduce-overhead"}


def test_gradient_checkpointing_is_on_unless_the_config_says_otherwise(workspace: Workspace) -> None:
    workspace.add_images("p", 1)
    config = workspace.config()
    del config["train"]["gradient_checkpointing"]

    lora_config = workspace.lora_config()
    del lora_config["train"]["gradient_checkpointing"]

    tuned, adapters = prepared_finetune(config), prepared_lora(lora_config)
    plain = prepared_lora(workspace.lora_config())

    assert tuned.transformer.gradient_checkpointing and adapters.transformer.gradient_checkpointing
    assert not plain.transformer.gradient_checkpointing


def test_checkpointing_the_blocks_changes_memory_use_and_not_the_trained_weights(workspace: Workspace) -> None:
    workspace.add_images("p", 2)
    plain = run_finetune(workspace.config(output={"path": str(workspace.root / "plain")}))

    checkpointed = run_finetune(workspace.config(train={"gradient_checkpointing": True}))

    assert checkpointed.transformer.gradient_checkpointing
    assert same_values(named_weights(checkpointed.transformer), named_weights(plain.transformer))


@pytest.mark.parametrize("builder", ["finetune", "lora"])
def test_compile_hands_the_blocks_to_torch_compile_after_the_model_is_built(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, builder: str
) -> None:
    workspace.add_images("p", 1)
    seen: list[tuple[int, torch.device, str | None]] = []

    def compile_spy(blocks: Any, device: torch.device, mode: str | None = None) -> bool:
        seen.append((len(list(blocks)), device, mode))
        return True

    monkeypatch.setattr(lora, "compile_blocks", compile_spy)

    if builder == "finetune":
        trainer: Any = prepared_finetune(workspace.config(train=COMPILE))
    else:
        trainer = prepared_lora(workspace.lora_config(train=COMPILE))

    assert seen == [(2, torch.device("cpu"), "reduce-overhead")]
    assert trainer.run_metadata["compiled"] is True


def test_without_compile_the_run_records_that_nothing_was_compiled(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    assert prepared_finetune(workspace.config()).run_metadata["compiled"] is False


@pytest.mark.parametrize("module", [finetune, lora], ids=["finetune", "lora"])
def test_a_dry_run_needs_no_output_section_and_writes_into_the_working_folder(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, module: Any
) -> None:
    workspace.add_images("p", 1)
    config = workspace.config()
    del config["output"]
    (workspace.root / "run.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.chdir(workspace.root)
    monkeypatch.setattr(sys, "argv", [module.__file__, "--config", "run.yaml", "--dry-run", "--dry-run-step-time", "0"])

    module.main()

    assert (workspace.root / "output" / "training.log").exists()
    assert not (workspace.root / "output" / "config.yaml").exists()
