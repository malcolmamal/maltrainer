"""Fizgig's training adapter over a finetune: on, it shapes what the weights train against; off, it is not read."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from tests.maltrainer.models.minimax_h3.h3_adapters import hand_pair, write_adapter
from tests.trainers.minimax_h3.world import Workspace

from .world import NAME, finetune_config, run_finetune


def _adapter(path: Path) -> Path:
    modules = {f"blocks.{n}.mlp.fc1": (*hand_pair(16, 16, rank=2, seed=n), 2.0) for n in range(3)}
    return write_adapter(path, modules)


def _run(ws: Workspace, adapter: Path | None, **finetune: Any) -> Any:
    ws.add_images(2)
    model = {"training_adapter_path": str(adapter)} if adapter is not None else {}
    config = finetune_config(ws, model=model, output={"name": NAME}, train={"max_steps": 2})
    config["finetune"] = {**config["finetune"], **finetune}
    return run_finetune(config)


def test_switched_on_the_adapter_rides_over_the_linears_and_is_recorded(workspace: Workspace) -> None:
    trainer = _run(workspace, _adapter(workspace.root / "adapter.safetensors"), training_adapter=True)

    assert sorted(trainer.adapter_layers) == ["blocks_0_mlp_fc1", "blocks_1_mlp_fc1", "blocks_2_mlp_fc1"]
    assert trainer.run_metadata["training_adapter_on"] is True and trainer.run_metadata["training_adapter_modules"] == 3
    assert not any(p.requires_grad for p in trainer.adapter_layers.parameters())


def test_the_adapter_changes_what_the_blocks_see_and_the_saved_model_has_no_trace_of_it(workspace: Workspace) -> None:
    adapter = _adapter(workspace.root / "adapter.safetensors")
    on = _run(workspace, adapter, training_adapter=True)
    on_file = load_file(str(workspace.out / f"{NAME}.safetensors"))
    off = _run(workspace, adapter, training_adapter=False)
    off_file = load_file(str(workspace.out / f"{NAME}.safetensors"))

    assert on.run_metadata["training_adapter_on"] and not off.run_metadata["training_adapter_on"]
    assert on_file.keys() == off_file.keys() and not any("lora" in key for key in on_file)
    assert not torch.equal(on_file["blocks.0.attn.qkv_proj.weight"], off_file["blocks.0.attn.qkv_proj.weight"])


def test_switched_off_the_adapter_file_is_not_even_read(workspace: Workspace) -> None:
    trainer = _run(workspace, workspace.root / "never_written.safetensors", training_adapter=False)

    assert len(trainer.adapter_layers) == 0 and trainer.run_metadata["training_adapter_on"] is False


def test_with_no_adapter_path_the_switch_changes_nothing(workspace: Workspace) -> None:
    trainer = _run(workspace, None, training_adapter=True)

    assert len(trainer.adapter_layers) == 0 and trainer.run_metadata["training_adapter_on"] is False


def test_an_adapter_for_other_modules_is_refused_by_name(workspace: Workspace) -> None:
    wrong = write_adapter(workspace.root / "wrong.safetensors", {"blocks.9.mlp.fc1": (*hand_pair(16, 16), 2.0)})

    with pytest.raises(ValueError, match="does not expose"):
        _run(workspace, wrong, training_adapter=True)
