"""A LoRA run on the tiny model: only the adapters train, and the saved file is the LoRA ComfyUI loads."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

import train_moss_image_lora as lora
from maltrainer.utils.lora_metadata import read_metadata

from .runs import run_lora, same_values
from .world import Workspace, base_weights, named_weights

BLOCK_LEAVES = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


def _keys(*, layers: tuple[int, ...] = (0, 1), leaves: tuple[str, ...] = BLOCK_LEAVES) -> set[str]:
    return {
        f"diffusion_model.layers.{i}.proj.{leaf}.{part}"
        for i in layers
        for leaf in leaves
        for part in ("lora_A.weight", "lora_B.weight", "alpha")
    }


def _modules(keys: set[str]) -> set[str]:
    return {key.removesuffix(".alpha").removesuffix(".lora_A.weight").removesuffix(".lora_B.weight") for key in keys}


def _trained(ws: Workspace, **sections: Any) -> tuple[lora.MOSSImageLoRATrainer, dict[str, torch.Tensor]]:
    ws.add_images("p", 2)
    trainer = run_lora(ws.lora_config(**sections))
    return trainer, load_file(str(ws.out / "moss_image_lora.safetensors"))


def test_only_the_adapters_train_and_the_base_stays_bit_identical(workspace: Workspace) -> None:
    trainer, saved = _trained(workspace)

    assert same_values(named_weights(trainer.transformer), base_weights(workspace))
    assert all(torch.count_nonzero(saved[key]) > 0 for key in saved if key.endswith("lora_B.weight"))


def test_the_saved_file_holds_one_a_b_and_alpha_per_adapter_in_bf16(workspace: Workspace) -> None:
    _, saved = _trained(workspace)

    assert set(saved) == _keys()
    assert saved["diffusion_model.layers.0.proj.q.lora_A.weight"].shape == (4, 128)
    assert saved["diffusion_model.layers.0.proj.ffn_down.lora_B.weight"].shape == (128, 4)
    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.bfloat16}
    assert saved["diffusion_model.layers.1.proj.o.alpha"].item() == 4.0


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        ("attn", _keys(leaves=("q", "k", "v", "o"))),
        ("blocks", _keys()),
    ],
)
def test_lora_targets_picks_which_linears_get_an_adapter(
    workspace: Workspace, targets: str, expected: set[str]
) -> None:
    _, saved = _trained(workspace, lora={"targets": targets})

    assert set(saved) == expected


def test_targets_all_adds_the_input_timestep_and_output_linears(workspace: Workspace) -> None:
    _, saved = _trained(workspace, lora={"targets": "all"})

    assert _modules(set(saved)) - _modules(_keys()) == {
        f"diffusion_model.{name}"
        for name in (
            "video_embedder",
            "time_embedding.0",
            "time_embedding.2",
            "shared_block_modulation",
            "video_head_modulation",
            "final_linear_video",
        )
    }


def test_rank_alpha_prefix_and_skipped_blocks_shape_the_adapters(workspace: Workspace) -> None:
    trainer, saved = _trained(workspace, lora={"rank": 8, "alpha": 2.5, "prefix": "unet.", "skip_blocks": [0]})

    assert {key.split(".layers.")[0] for key in saved} == {"unet"}
    assert {key.split(".")[2] for key in saved} == {"1"}
    assert saved["unet.layers.1.proj.q.lora_A.weight"].shape == (8, 128)
    assert saved["unet.layers.1.proj.q.alpha"].item() == 2.5
    assert trainer.run_metadata["lora_skip_blocks"] == [0]


def test_an_unknown_target_preset_is_refused(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    with pytest.raises(ValueError, match=r"lora\.targets must be one of \('blocks', 'attn', 'all'\), got 'mlp'"):
        run_lora(workspace.lora_config(lora={"targets": "mlp"}))


def test_the_lora_dtype_and_the_output_dtype_are_independent(workspace: Workspace) -> None:
    trainer, saved = _trained(workspace, lora={"dtype": "bfloat16"}, output={"dtype": "float32"})

    assert trainer.run_metadata["lora_dtype"] == "bfloat16"
    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.float32}


def test_metadata_records_the_adapters_the_optimizer_and_the_config(workspace: Workspace) -> None:
    trainer, _ = _trained(workspace, train={"ema_decay": 0.5})

    meta = read_metadata(workspace.out / "moss_image_lora.safetensors")
    assert (meta["maltrainer_architecture"], meta["maltrainer_trainer"]) == ("moss_image", "train_moss_image_lora.py")
    assert (meta["ss_network_dim"], meta["ss_network_alpha"], meta["step"]) == ("4", "4", "final")
    assert (meta["maltrainer_lora_layers"], meta["maltrainer_lora_dtype"], meta["maltrainer_compiled"]) == (
        "12",
        "float32",
        "false",
    )
    assert meta["maltrainer_int8_grad_precision"] == "bf16" and meta["maltrainer_ema_decay"] == "0.5"
    assert json.loads(meta["maltrainer_optimizer_resolved"])["name"] == "automagic3"
    assert int(meta["maltrainer_trainable_parameters"]) == trainer.run_metadata["trainable_parameters"]
    assert json.loads(meta["maltrainer_config"])["lora"] == {"rank": 4, "alpha": 4}


def test_with_ema_the_checkpoint_holds_the_average_and_the_raw_adapters_come_back(workspace: Workspace) -> None:
    trainer, saved = _trained(workspace, train={"ema_decay": 0.5, "ema_warmup": False})

    assert trainer.ema is not None
    average = trainer.ema.shadow[1].to(torch.bfloat16)
    assert torch.equal(saved["diffusion_model.layers.0.proj.q.lora_B.weight"], average)
    raw = trainer.accelerator.unwrap_model(trainer.lora_network).get_state_dict(torch.bfloat16)
    assert not torch.equal(raw["diffusion_model.layers.0.proj.q.lora_B.weight"], average)
