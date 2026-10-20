"""A whole H3 run on the tiny world: what trains, what is saved, and that a run repeats exactly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from accelerate.state import AcceleratorState, GradientState
from safetensors.torch import load_file

import train_minimax_h3_lora as h3
from maltrainer.utils.lora_metadata import read_metadata
from tests.maltrainer.models.minimax_h3.h3_adapters import hand_pair, write_adapter

from .fakes import Fakes
from .runs import run_h3
from .world import Workspace, named_weights, same_tensors

FINAL = "minimax_h3_lora_step_final.safetensors"
PROJECTIONS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
MODULES = [f"blocks.{i}.{p}" for i in range(3) for p in PROJECTIONS] + [
    f"token_refiner.blocks.0.{p}" for p in PROJECTIONS
]


def _mixed(ws: Workspace, **sections: Any) -> tuple[h3.MiniMaxH3LoRATrainer, dict[str, torch.Tensor]]:
    ws.add_images(2)
    ws.add_clip()
    ws.add_voice()
    trainer = run_h3(ws.config(**sections))
    return trainer, load_file(str(ws.out / FINAL))


def test_only_the_adapters_train_and_the_base_stays_bit_identical(workspace: Workspace, fakes: Fakes) -> None:
    trainer, saved = _mixed(workspace)

    assert same_tensors(named_weights(trainer.transformer), fakes.reference_state())
    ups = [key for key in saved if key.endswith("lora_B.weight")]
    assert len(ups) == 16 and all(torch.count_nonzero(saved[key]) > 0 for key in ups)


def test_a_pruned_adaln_checkpoint_trains_the_same_way_and_keeps_its_table(workspace: Workspace, fakes: Fakes) -> None:
    fakes.adaln_table = 4

    trainer, saved = _mixed(workspace)

    assert same_tensors(named_weights(trainer.transformer), fakes.reference_state())
    assert len(saved) == 48 and trainer.transformer.time_embedder is None


def test_the_saved_file_holds_one_a_b_and_alpha_per_adapter_in_bf16(workspace: Workspace) -> None:
    _, saved = _mixed(workspace, lora={"rank": 2, "alpha": 6})

    assert set(saved) == {
        f"diffusion_model.{m}.{part}" for m in MODULES for part in ("lora_A.weight", "lora_B.weight", "alpha")
    }
    assert saved["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"].shape == (2, 16)
    assert saved["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"].shape == (48, 2)
    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.bfloat16}
    assert saved["diffusion_model.blocks.2.mlp.fc2.alpha"].item() == 6.0


def test_the_output_dtype_is_independent_of_the_training_dtypes(workspace: Workspace) -> None:
    _, saved = _mixed(workspace, output={"dtype": "float32"}, dtype="bfloat16", lora={"dtype": "bfloat16"})

    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.float32}


def test_metadata_records_the_run_the_adapters_and_the_resolved_settings(workspace: Workspace) -> None:
    trainer, _ = _mixed(workspace, train={"ema_decay": 0.5})

    meta = read_metadata(workspace.out / FINAL)
    assert (meta["maltrainer_architecture"], meta["maltrainer_trainer"]) == ("minimax_h3", "train_minimax_h3_lora.py")
    assert (meta["ss_network_dim"], meta["ss_network_alpha"], meta["step"]) == ("4", "4", "final")
    assert (meta["maltrainer_lora_layers"], meta["maltrainer_lora_dtype"], meta["maltrainer_save_dtype"]) == (
        "16",
        "float32",
        "bfloat16",
    )
    assert (meta["maltrainer_git_commit"], meta["maltrainer_ema_decay"]) == ("abc1234", "0.5")
    assert json.loads(meta["maltrainer_dataset_composition"]) == {
        "audio_items": 1,
        "clips": 1,
        "clips_with_audio": 1,
        "images": 2,
    }
    assert int(meta["maltrainer_trainable_parameters"]) == trainer.run_metadata["trainable_parameters"] == 2432
    assert json.loads(meta["maltrainer_config"])["lora"] == {"rank": 4, "alpha": 4}


def test_embed_config_false_leaves_the_config_out_of_the_file(workspace: Workspace) -> None:
    _mixed(workspace, output={"embed_config": False})

    assert "maltrainer_config" not in read_metadata(workspace.out / FINAL)


def test_a_training_adapter_is_recorded_in_the_metadata(workspace: Workspace) -> None:
    down, up = hand_pair(16, 16)
    path = write_adapter(workspace.root / "assist.safetensors", {"blocks.0.mlp.fc1": (down, up, 2.0)})

    _mixed(workspace, model={"training_adapter_path": str(path)})

    meta = read_metadata(workspace.out / FINAL)
    assert (meta["maltrainer_training_adapter"], meta["maltrainer_training_adapter_modules"]) == (
        "assist.safetensors",
        "1",
    )


def test_with_ema_the_checkpoint_holds_the_average_while_the_live_adapters_stay_raw(workspace: Workspace) -> None:
    trainer, saved = _mixed(workspace, train={"ema_decay": 0.5, "ema_warmup": False})

    assert trainer.ema is not None and trainer.ema.updates == 4
    # the first adapter in module order is the token refiner's; its parameters go A, B
    average = trainer.ema.shadow[1].to(torch.bfloat16)
    assert torch.equal(saved["diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_B.weight"], average)
    raw = trainer.accelerator.unwrap_model(trainer.lora_network).get_state_dict(torch.bfloat16)
    assert not torch.equal(raw["diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_B.weight"], average)


@pytest.mark.parametrize("decay", [0, None, 0.0])
def test_without_a_decay_there_is_no_ema(workspace: Workspace, decay: float | None) -> None:
    trainer, _ = _mixed(workspace, train={"ema_decay": decay})

    assert trainer.ema is None and "ema_decay" not in trainer.run_metadata


def _again(ws: Workspace, seed: int, out: Path) -> dict[str, torch.Tensor]:
    AcceleratorState._reset_state(True)
    GradientState._reset_state()
    run_h3(ws.config(seed=seed, output={"path": str(out)}))
    return load_file(str(out / FINAL))


def test_a_run_repeats_bit_for_bit_with_the_same_seed_and_differs_with_another(workspace: Workspace) -> None:
    workspace.add_images(2)
    workspace.add_clip()
    workspace.add_voice()

    first = _again(workspace, 7, workspace.root / "one")
    second = _again(workspace, 7, workspace.root / "two")
    other = _again(workspace, 8, workspace.root / "three")

    assert same_tensors(first, second) and not same_tensors(first, other)
