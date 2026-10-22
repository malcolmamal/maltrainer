"""Loading the base transformer and injecting the adapters, with and without a training adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

import train_qwen_image_2_1_lora as qwen

from .fakes import Fakes
from .runs import loaded, lora_layer
from .world import Workspace, same_tensors

Q0 = "transformer_blocks.0.attn.to_q"


def _loaded(ws: Workspace, **sections: Any) -> qwen.QwenImage21LoRATrainer:
    ws.add_images(1)
    return loaded(ws.config(**sections))


def _layers(trainer: qwen.QwenImage21LoRATrainer) -> list[str]:
    return sorted(trainer.lora_network.name_map.values())


def test_the_transformer_loader_gets_the_path_device_dtype_and_gradient_precision(
    workspace: Workspace, fakes: Fakes
) -> None:
    trainer = _loaded(workspace, dtype="bfloat16", train={"int8_grad_precision": "fp32"})

    assert (fakes.transformer_paths, fakes.dtypes, fakes.grad_precisions) == (
        ["transformer.safetensors"],
        [torch.bfloat16],
        ["fp32"],
    )
    assert (
        trainer.run_metadata["int8_grad_precision"] == "fp32" and trainer.run_metadata["model_type"] == "qwen_image_2_1"
    )


def test_the_gradient_precision_defaults_to_bf16(workspace: Workspace, fakes: Fakes) -> None:
    trainer = _loaded(workspace)

    assert fakes.grad_precisions == ["bf16"] and trainer.run_metadata["int8_grad_precision"] == "bf16"


def test_the_base_stays_frozen_and_untouched_by_the_adapter_injection(workspace: Workspace, fakes: Fakes) -> None:
    trainer = _loaded(workspace)

    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    assert same_tensors(
        {k: v.detach().clone() for k, v in trainer.transformer.state_dict().items()}, fakes.reference_state()
    )


def test_by_default_the_attention_and_mlp_linears_of_every_block_get_adapters(workspace: Workspace) -> None:
    trainer = _loaded(workspace)

    names = _layers(trainer)
    assert len(names) == 12 and names[0] == "transformer_blocks.0.attn.to_k"
    assert {n.split(".", 2)[2] for n in names} == {
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
        "img_mlp.gate_up",
        "img_mlp.out",
    }
    assert trainer.run_metadata["lora_layers"] == 12 and trainer.run_metadata["trainable_parameters"] == 2048
    assert trainer.run_metadata["lora_dtype"] == "float32"


@pytest.mark.parametrize(
    ("targets", "expected"),
    [(["to_q"], 2), (["to_q", "img_mlp.out"], 4), (["to_out.0"], 2), ([], 12)],
)
def test_target_modules_pick_the_linears_by_the_end_of_their_name(
    workspace: Workspace, targets: list[str], expected: int
) -> None:
    assert len(_layers(_loaded(workspace, lora={"target_modules": targets}))) == expected


def test_rank_alpha_prefix_and_dtype_shape_the_adapters(workspace: Workspace) -> None:
    trainer = _loaded(workspace, lora={"rank": 2, "alpha": 8.0, "prefix": "unet.", "dtype": "bfloat16"})

    layer = lora_layer(trainer, "transformer_blocks_0_attn_to_q")
    assert (layer.lora_down.weight.shape, layer.lora_up.weight.shape) == ((2, 16), (16, 2))
    assert layer.scale == 4.0 and trainer.lora_network.prefix == "unet."
    assert {p.dtype for p in trainer.lora_network.trainable_parameters()} == {torch.bfloat16}
    assert trainer.run_metadata["lora_dtype"] == "bfloat16"


def _assistant(ws: Workspace) -> Path:
    path = ws.root / "assistant.safetensors"
    save_file(
        {
            f"diffusion_model.{Q0}.lora_A.weight": torch.full((2, 16), 0.05),
            f"diffusion_model.{Q0}.lora_B.weight": torch.full((16, 2), 0.5),
            f"diffusion_model.{Q0}.alpha": torch.tensor(4.0),
        },
        str(path),
    )
    return path


@pytest.mark.parametrize("where", ["model", "lora"])
def test_a_training_adapter_is_attached_live_and_adds_no_trainable_parameters(workspace: Workspace, where: str) -> None:
    trainer = _loaded(workspace, **{where: {"training_adapter_path": str(_assistant(workspace))}})

    assert trainer.run_metadata["training_adapter"] == "assistant.safetensors"
    assert trainer.run_metadata["training_adapter_modules"] == 1
    assert trainer.run_metadata["trainable_parameters"] == 2048


def test_the_training_adapter_changes_the_base_output_by_its_scaled_delta_and_the_multiplier_scales_it(
    workspace: Workspace,
) -> None:
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(2))
    plain = _loaded(workspace)
    base = plain.transformer.get_submodule(Q0)(x)
    path = _assistant(workspace)

    outputs = []
    for index, multiplier in enumerate((1.0, 0.5)):
        sub = Workspace(workspace.root / f"w{index}")
        trainer = _loaded(
            sub, model={"training_adapter_path": str(path)}, lora={"training_adapter_multiplier": multiplier}
        )
        outputs.append(trainer.transformer.get_submodule(Q0)(x))

    delta = (x @ torch.full((16, 2), 0.05)) @ torch.full((2, 16), 0.5) * 2.0  # alpha 4 / rank 2
    torch.testing.assert_close(outputs[0] - base, delta, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(outputs[1] - base, delta * 0.5, atol=1e-5, rtol=1e-4)
