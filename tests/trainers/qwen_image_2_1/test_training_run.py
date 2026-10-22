"""A whole Qwen run on the tiny world: what trains, what is saved and under which name, and that a run repeats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

import train_qwen_image_2_1_lora as qwen
from maltrainer.utils.lora_metadata import read_metadata

from .fakes import Fakes
from .runs import rerun, run_qwen
from .world import Workspace, named_weights, same_tensors

FINAL = "qwen_image_2_1_lora.safetensors"
LINEARS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out")
MODULES = [f"transformer_blocks.{i}.{leaf}" for i in range(2) for leaf in LINEARS]


def _trained(ws: Workspace, **sections: Any) -> tuple[qwen.QwenImage21LoRATrainer, dict[str, torch.Tensor]]:
    ws.add_images(2)
    trainer = run_qwen(ws.config(**sections))
    return trainer, load_file(str(ws.out / FINAL))


def test_only_the_adapters_train_and_the_base_stays_bit_identical(workspace: Workspace, fakes: Fakes) -> None:
    trainer, saved = _trained(workspace)

    assert same_tensors(named_weights(trainer.transformer), fakes.reference_state())
    ups = [key for key in saved if key.endswith("lora_B.weight")]
    assert len(ups) == 12 and all(torch.count_nonzero(saved[key]) > 0 for key in ups)


def test_the_saved_file_holds_one_a_b_and_alpha_per_adapter_in_bf16(workspace: Workspace) -> None:
    _, saved = _trained(workspace, lora={"rank": 2, "alpha": 6})

    assert set(saved) == {
        f"diffusion_model.{m}.{part}" for m in MODULES for part in ("lora_A.weight", "lora_B.weight", "alpha")
    }
    assert saved["diffusion_model.transformer_blocks.0.attn.to_q.lora_A.weight"].shape == (2, 16)
    assert saved["diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight"].shape == (64, 2)
    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.bfloat16}
    assert saved["diffusion_model.transformer_blocks.1.img_mlp.out.alpha"].item() == 6.0


@pytest.mark.parametrize(("fmt", "names"), [("kohya", ("lora_down", "lora_up")), ("Diffusers", ("lora_A", "lora_B"))])
def test_the_output_format_picks_the_adapter_key_names(workspace: Workspace, fmt: str, names: tuple[str, str]) -> None:
    _, saved = _trained(workspace, output={"format": fmt})

    assert {key.split(".")[-2] for key in saved if not key.endswith("alpha")} == set(names)
    assert len(saved) == 36


def test_the_output_dtype_is_independent_of_the_training_dtypes(workspace: Workspace) -> None:
    _, saved = _trained(workspace, output={"dtype": "float32"}, dtype="bfloat16", lora={"dtype": "bfloat16"})

    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.float32}


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ({"name": "mine"}, "mine.safetensors"),
        ({"name": "mine.safetensors"}, "mine.safetensors"),
        ({"lora_name": "legacy"}, "legacy.safetensors"),
        ({"file_name": "older.safetensors"}, "older.safetensors"),
        ({"name": "first", "lora_name": "second"}, "first.safetensors"),
        ({}, FINAL),
    ],
)
def test_the_checkpoint_name_comes_from_name_then_lora_name_then_file_name_then_the_default(
    workspace: Workspace, output: dict[str, str], expected: str
) -> None:
    workspace.add_images(1)

    run_qwen(workspace.config(train={"max_steps": 1}, output=output))

    assert sorted(path.name for path in workspace.out.glob("*.safetensors")) == [expected]


def test_a_named_run_numbers_its_intermediate_checkpoints_after_the_name(workspace: Workspace) -> None:
    workspace.add_images(1)

    run_qwen(workspace.config(train={"max_steps": 3, "save_every": 2}, output={"name": "mine.safetensors"}))

    assert sorted(path.name for path in workspace.out.glob("*.safetensors")) == [
        "mine.safetensors",
        "mine_step_2.safetensors",
    ]


def test_metadata_records_the_run_the_adapters_and_the_resolved_settings(workspace: Workspace) -> None:
    trainer, _ = _trained(workspace)

    meta = read_metadata(workspace.out / FINAL)
    assert (meta["maltrainer_architecture"], meta["maltrainer_trainer"]) == (
        "qwen_image_2_1",
        "train_qwen_image_2_1_lora.py",
    )
    assert (meta["ss_network_dim"], meta["ss_network_alpha"], meta["step"]) == ("4", "4", "final")
    assert (meta["maltrainer_lora_layers"], meta["maltrainer_lora_dtype"], meta["maltrainer_save_dtype"]) == (
        "12",
        "float32",
        "bfloat16",
    )
    assert (meta["maltrainer_git_commit"], meta["maltrainer_int8_grad_precision"]) == ("abc1234", "bf16")
    assert json.loads(meta["maltrainer_optimizer_resolved"]) == {"name": "adamw", "lr": 1e-3}
    assert int(meta["maltrainer_trainable_parameters"]) == trainer.run_metadata["trainable_parameters"] == 2048
    assert json.loads(meta["maltrainer_config"])["lora"] == {"rank": 4, "alpha": 4}


def test_embed_config_false_leaves_the_config_out_of_the_file(workspace: Workspace) -> None:
    _trained(workspace, output={"embed_config": False})

    assert "maltrainer_config" not in read_metadata(workspace.out / FINAL)


def test_a_training_adapter_is_recorded_in_the_metadata_and_the_run_still_trains_only_the_lora(
    workspace: Workspace, fakes: Fakes
) -> None:
    path = workspace.root / "assist.safetensors"
    q = "transformer_blocks.0.attn.to_q"
    save_file(
        {
            f"diffusion_model.{q}.lora_A.weight": torch.full((2, 16), 0.05),
            f"diffusion_model.{q}.lora_B.weight": torch.full((16, 2), 0.5),
        },
        str(path),
    )

    trainer, saved = _trained(workspace, model={"training_adapter_path": str(path)})

    meta = read_metadata(workspace.out / FINAL)
    assert (meta["maltrainer_training_adapter"], meta["maltrainer_training_adapter_modules"]) == (
        "assist.safetensors",
        "1",
    )
    assert len(saved) == 36 and same_tensors(named_weights(trainer.transformer), fakes.reference_state())


def _again(ws: Workspace, seed: int, out: Path) -> dict[str, torch.Tensor]:
    rerun(ws.config(seed=seed, output={"path": str(out)}))
    return load_file(str(out / FINAL))


def test_a_run_repeats_bit_for_bit_with_the_same_seed_and_differs_with_another(workspace: Workspace) -> None:
    workspace.add_images(3)

    first = _again(workspace, 7, workspace.root / "one")
    second = _again(workspace, 7, workspace.root / "two")
    other = _again(workspace, 8, workspace.root / "three")

    assert same_tensors(first, second) and not same_tensors(first, other)
