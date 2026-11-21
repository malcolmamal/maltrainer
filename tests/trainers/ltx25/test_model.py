"""Step 2 of a run: the frozen DiT from the checkpoint, which adapters it gets and what it refuses."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from .runs import prepared, run_lora
from .world import HEADS, Workspace, named_weights


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
        "file": workspace.checkpoint.name,
        "format": "float32",
        "int8_layers": 0,
        "layers": 2,
        "audio": False,
    }


def test_images_only_train_a_video_model_with_the_video_adapters(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared(workspace.config())

    assert not trainer.transformer.with_audio
    assert trainer.run_metadata["lora_layers"] == 16  # 2 blocks x (attn1, attn2) x 4 projections


def test_clips_with_sound_train_the_audio_stream_and_all_attention(workspace: Workspace) -> None:
    workspace.add_clip("loud", seconds=0.5)

    trainer = prepared(workspace.config())

    assert trainer.transformer.with_audio and trainer.run_metadata["lora_layers"] == 48


def test_the_lora_targets_can_be_chosen(workspace: Workspace) -> None:
    workspace.add_images(1)

    video_ff = prepared(workspace.config(lora={"targets": "video_ff"}))
    full = prepared(workspace.config(lora={"targets": "full"}))

    assert video_ff.run_metadata["lora_layers"] == 20
    assert full.run_metadata["lora_layers"] > video_ff.run_metadata["lora_layers"]


def test_a_missing_checkpoint_is_reported_by_the_loader(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared(workspace.config())
    trainer.config["model"]["transformer_path"] = str(workspace.root / "gone.safetensors")

    with pytest.raises(FileNotFoundError, match=r"LTX-2\.5 transformer not found at .*gone\.safetensors"):
        trainer.load_transformer_and_lora()


def test_a_distilled_checkpoint_stops_the_run_with_the_reason(workspace: Workspace) -> None:
    distilled = workspace.root / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    save_file(load_file(str(workspace.checkpoint)), str(distilled))
    workspace.add_images(1)

    with pytest.raises(ValueError, match="train the dev model"):
        run_lora(workspace.config(model={"transformer_path": str(distilled)}))


def test_the_metadata_heads_reach_the_model(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared(workspace.config())

    assert trainer.transformer.config.num_heads == HEADS["num_attention_heads"]
