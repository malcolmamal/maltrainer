"""Step 2 of a run: the frozen DiT from the checkpoint, gradient checkpointing and the int8 ConvRot path."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.maltrainer.models.flux3.checkpoints import int8_state, write

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
    assert trainer.run_metadata["base_load"] == {
        "file": "klein.safetensors",
        "format": "float32",
        "int8_layers": 0,
        "hidden_size": 128,
        "depth": 1,
        "depth_single_blocks": 2,
        "fused_qkv": False,
        "inference_attention": {},
    }
    assert trainer.run_metadata["lora_layers"] == 24


def test_a_missing_checkpoint_is_reported_by_the_loader(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared(workspace.config())
    trainer.config["model"]["transformer_path"] = str(workspace.root / "gone.safetensors")

    with pytest.raises(FileNotFoundError, match=r"FLUX\.3 transformer not found at .*gone\.safetensors"):
        trainer.load_transformer_and_lora()


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_an_int8_convrot_checkpoint_trains_through_the_frozen_int8_linears(
    workspace: Workspace, precision: str
) -> None:
    write(workspace.checkpoint, int8_state())
    workspace.add_images(2)

    trainer = run_lora(workspace.config(train={"int8_grad_precision": precision}))

    assert trainer.run_metadata["int8_grad_precision"] == precision
    base = trainer.run_metadata["base_load"]
    assert isinstance(base, dict) and base["format"] == "int8_convrot" and base["int8_layers"] > 0
    assert (workspace.out / "klein.safetensors").exists()
