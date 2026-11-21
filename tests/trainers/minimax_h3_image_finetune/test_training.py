"""A whole H3 Image finetune on the tiny world: windows rotate, the file loads as INT8, the rest is untouched."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from maltrainer.models.minimax_h3.int8_convrot import load_comfy_int8_convrot_state_dict
from maltrainer.utils.lora_metadata import read_metadata
from tests.trainers.minimax_h3.world import Workspace

from .world import NAME, finetune_config, run_finetune


def _run(ws: Workspace, images: int = 2, **sections: Any) -> Any:
    ws.add_images(images)
    return run_finetune(finetune_config(ws, output={"name": NAME}, **sections))


def test_the_final_file_loads_as_an_int8_convrot_model_with_the_base_files_tensors(workspace: Workspace) -> None:
    _run(workspace, train={"max_steps": 3})

    saved, count = load_comfy_int8_convrot_state_dict(
        str(workspace.out / f"{NAME}.safetensors"), device=torch.device("cpu")
    )
    base = load_file(str(workspace.root / "transformer.safetensors"))
    assert count == 12 and load_file(str(workspace.out / f"{NAME}.safetensors")).keys() == base.keys()
    assert saved["blocks.0.attn.qkv_proj.weight"].dtype is torch.int8


def test_every_window_trains_in_turn_and_only_the_trained_blocks_change(workspace: Workspace) -> None:
    trainer = _run(workspace, train={"max_steps": 2}, finetune={"block_range": [0, 1]})

    saved, base = (
        load_file(str(workspace.out / f"{NAME}.safetensors")),
        load_file(str(workspace.root / "transformer.safetensors")),
    )
    changed = {k for k in base if not torch.equal(saved[k], base[k])}
    assert trainer.rotator.names == ["blocks 0-0", "blocks 1-1"]
    assert {k for k in changed if k.endswith(".weight")} == {
        f"blocks.{n}.{leaf}.weight" for n in (0, 1) for leaf in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    }
    assert not any(k.startswith("blocks.2.") or "refiner" in k for k in changed)


def test_windows_come_up_in_order_every_rotate_every_steps(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []
    from maltrainer.finetune.rotation_int8 import Int8WindowRotator

    original = Int8WindowRotator.activate
    monkeypatch.setattr(
        Int8WindowRotator, "activate", lambda self, index: (seen.append(index), original(self, index))[1]
    )

    _run(workspace, train={"max_steps": 6}, finetune={"rotate_every": 2})

    assert seen == [0, 1, 2]


def test_always_on_modules_train_beside_the_windows_and_are_saved_in_their_own_dtype(workspace: Workspace) -> None:
    _run(workspace, train={"max_steps": 2}, finetune={"always_on": ["final_layer.video_out"]})

    saved, base = (
        load_file(str(workspace.out / f"{NAME}.safetensors")),
        load_file(str(workspace.root / "transformer.safetensors")),
    )
    assert saved["final_layer.video_out.weight"].dtype is base["final_layer.video_out.weight"].dtype
    assert not torch.equal(saved["final_layer.video_out.weight"], base["final_layer.video_out.weight"])


def test_the_file_records_the_run_and_keeps_the_noise_level_grid(workspace: Workspace) -> None:
    _run(workspace, train={"max_steps": 1})

    meta = read_metadata(workspace.out / f"{NAME}.safetensors")
    assert meta["maltrainer_architecture"] == "minimax_h3_image_finetune"
    assert meta["maltrainer_trainer"] == "train_minimax_h3_image_finetune.py"
    assert '"pdd_sigmas"' in meta["config"]
