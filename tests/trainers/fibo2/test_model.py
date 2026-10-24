"""Step 2 of a run: the frozen DiT from the checkpoint, gradient checkpointing and the checkpoints it refuses."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from .runs import prepared, run_lora
from .world import Workspace, named_weights


def _adapters(ws: Workspace, **sections: Any) -> dict[str, torch.Tensor]:
    ws.add_images(2)
    return named_weights(run_lora(ws.config(**sections)).lora_network)


def test_gradient_checkpointing_is_on_by_default_and_changes_no_number(workspace: Workspace) -> None:
    workspace.add_images(1)
    config = workspace.config()
    del config["train"]["gradient_checkpointing"]
    on = prepared(config)
    off = prepared(workspace.config())

    assert on.transformer.gradient_checkpointing and not off.transformer.gradient_checkpointing
    checkpointed = _adapters(
        workspace, output={"path": str(workspace.root / "a")}, train={"gradient_checkpointing": True}
    )
    plain = _adapters(workspace, output={"path": str(workspace.root / "b")})
    for name, value in plain.items():
        torch.testing.assert_close(checkpointed[name], value)


def test_the_loaded_dit_is_frozen_in_eval_mode_in_the_training_dtype_and_summarised(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared(workspace.config(dtype="bfloat16"))

    assert not trainer.transformer.training and trainer.transformer.compute_dtype == torch.bfloat16
    assert {p.dtype for p in trainer.transformer.parameters()} == {torch.bfloat16}
    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    assert trainer.run_metadata["base_load"] == {
        "file": "fibo2.safetensors",
        "format": "float32",
        "dim": 256,
        "layers": 6,
        "injection_layers": [1, 2, 3, 4, 5],
    }
    assert trainer.run_metadata["lora_layers"] == 43


def test_a_missing_checkpoint_is_reported_by_the_loader(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared(workspace.config())
    trainer.config["model"]["transformer_path"] = str(workspace.root / "gone.safetensors")

    with pytest.raises(FileNotFoundError, match=r"Fibo 2\.0 transformer not found at .*gone\.safetensors"):
        trainer.load_transformer_and_lora()


def test_an_int8_checkpoint_stops_the_run_with_the_reason(workspace: Workspace) -> None:
    state = load_file(str(workspace.checkpoint))
    state["layers.0.attention.qkv.comfy_quant"] = torch.zeros(4, dtype=torch.uint8)
    save_file(state, str(workspace.checkpoint))
    workspace.add_images(1)

    with pytest.raises(ValueError, match="Edit Turbo"):
        run_lora(workspace.config())
