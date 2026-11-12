"""Step 2 of a run: the frozen DiT from the checkpoint, gradient checkpointing and the checkpoints it refuses."""

from __future__ import annotations

from typing import Any

import pytest
import torch

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


def test_an_edit_turbo_int8_checkpoint_stops_the_run_with_the_reason(workspace: Workspace) -> None:
    from tests.maltrainer.models.fibo2.int8_files import write_int8

    path, _ = write_int8(workspace.root / "fibo2_edit_turbo_int8_convrot.safetensors")
    workspace.add_images(1)

    with pytest.raises(ValueError, match="Edit Turbo"):
        run_lora(workspace.config(model={"transformer_path": str(path)}))


def test_a_dev_int8_checkpoint_trains_decoded_by_default(workspace: Workspace) -> None:
    from tests.maltrainer.models.fibo2.int8_files import write_int8

    path, stored = write_int8(workspace.root / "fibo2_dev_int8_convrot.safetensors")
    workspace.add_images(2)

    trainer = run_lora(workspace.config(model={"transformer_path": str(path)}))

    assert trainer.run_metadata["base_load"]["int8_load"] == "decode"
    assert trainer.run_metadata["base_load"]["int8_layers"] == len(stored)
    assert {p.dtype for p in trainer.transformer.parameters()} == {torch.float32}  # the tiny world trains in fp32


def test_model_int8_load_runtime_keeps_the_dev_int8_linears_int8(workspace: Workspace) -> None:
    from tests.maltrainer.models.fibo2.int8_files import write_int8

    path, stored = write_int8(workspace.root / "fibo2_dev_int8_convrot.safetensors")
    workspace.add_images(1)

    trainer = prepared(workspace.config(model={"transformer_path": str(path), "int8_load": "runtime"}))

    assert trainer.run_metadata["base_load"]["int8_load"] == "runtime"
    assert trainer.transformer.get_submodule(stored[0]).weight.dtype == torch.int8
