"""Step 2 of a run: the frozen DiT from the checkpoint and the checkpoints it refuses."""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file

from .runs import prepared, run_lora
from .world import Workspace


def test_the_loaded_dit_is_frozen_in_eval_mode_in_the_training_dtype_and_summarised(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared(workspace.config(dtype="bfloat16"))

    assert not trainer.transformer.training
    assert {p.dtype for p in trainer.transformer.parameters()} == {torch.bfloat16}
    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    assert trainer.run_metadata["base_load"] == {
        "file": "visage.safetensors",
        "format": "float32",
        "width": 128,
        "depth": 2,
        "latent_channels": 8,
    }
    assert trainer.run_metadata["lora_layers"] == 8


def test_a_missing_checkpoint_is_reported_by_the_loader(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared(workspace.config())
    trainer.config["model"]["transformer_path"] = str(workspace.root / "gone.safetensors")

    with pytest.raises(FileNotFoundError, match=r"Visage-Image transformer not found at .*gone\.safetensors"):
        trainer.load_transformer_and_lora()


def test_a_file_that_is_not_the_dit_stops_the_run_with_the_reason(workspace: Workspace) -> None:
    state = load_file(str(workspace.checkpoint))
    del state["text_registers"]
    save_file(state, str(workspace.checkpoint))
    workspace.add_images(1)

    with pytest.raises(ValueError, match="not a Visage-Image DiT checkpoint"):
        run_lora(workspace.config())
