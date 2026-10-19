from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRANetwork

from .test_lora_network import Toy

CPU = torch.device("cpu")


def _network() -> QwenImage21LoRANetwork:
    net = QwenImage21LoRANetwork(Toy(), rank=2, alpha=4.0, target_modules=["to_q"])
    for layer in net.lora_layers.values():
        nn.init.normal_(layer.lora_up.weight)
    return net


def test_diffusers_state_dict_uses_lora_a_b_keys_with_the_prefix_and_an_alpha_scalar() -> None:
    state = _network().get_state_dict()

    assert sorted(state) == [
        "diffusion_model.transformer_blocks.0.to_q.alpha",
        "diffusion_model.transformer_blocks.0.to_q.lora_A.weight",
        "diffusion_model.transformer_blocks.0.to_q.lora_B.weight",
        "diffusion_model.transformer_blocks.1.to_q.alpha",
        "diffusion_model.transformer_blocks.1.to_q.lora_A.weight",
        "diffusion_model.transformer_blocks.1.to_q.lora_B.weight",
    ]
    assert state["diffusion_model.transformer_blocks.0.to_q.alpha"].item() == 4.0
    assert state["diffusion_model.transformer_blocks.0.to_q.alpha"].dtype == torch.float32


def test_kohya_state_dict_uses_lora_down_up_keys_holding_the_same_weights() -> None:
    net = _network()

    state = net.get_state_dict(save_dtype=None, format_type="kohya")

    layer = net.lora_layers["transformer_blocks_1_to_q"]
    assert torch.equal(state["diffusion_model.transformer_blocks.1.to_q.lora_down.weight"], layer.lora_down.weight)
    assert torch.equal(state["diffusion_model.transformer_blocks.1.to_q.lora_up.weight"], layer.lora_up.weight)
    assert not any("lora_A" in key for key in state)


def test_saved_weights_are_cast_to_the_requested_dtype_and_detached_on_cpu() -> None:
    net = _network()

    bf16 = net.get_state_dict()
    native = net.get_state_dict(save_dtype=None)

    weight_key = "diffusion_model.transformer_blocks.0.to_q.lora_A.weight"
    assert bf16[weight_key].dtype == torch.bfloat16 and native[weight_key].dtype == torch.float32
    assert not native[weight_key].requires_grad and native[weight_key].device.type == "cpu"


def test_a_custom_prefix_is_used_in_every_saved_key() -> None:
    net = QwenImage21LoRANetwork(Toy(), rank=1, alpha=1.0, target_modules=["to_q"], prefix="lora_unet.")

    assert all(key.startswith("lora_unet.transformer_blocks.") for key in net.get_state_dict())


def _assistant_file(path: Path, rank: int = 2, keys: str = "diffusers", alpha: float | None = 8.0) -> dict:
    generator = torch.Generator().manual_seed(5)
    down_name, up_name = ("lora_A", "lora_B") if keys == "diffusers" else ("lora_down", "lora_up")
    base = "diffusion_model.transformer_blocks.0.to_q"
    tensors = {
        f"{base}.{down_name}.weight": torch.randn(rank, 4, generator=generator),
        f"{base}.{up_name}.weight": torch.randn(4, rank, generator=generator),
        # not a linear, not in the model, and half a pair: all three must be skipped
        "diffusion_model.transformer_blocks.0.to_out.1.lora_A.weight": torch.randn(rank, 4, generator=generator),
        "diffusion_model.transformer_blocks.0.to_out.1.lora_B.weight": torch.randn(4, rank, generator=generator),
        "diffusion_model.ghost.layer.lora_A.weight": torch.randn(rank, 4, generator=generator),
        "diffusion_model.ghost.layer.lora_B.weight": torch.randn(4, rank, generator=generator),
        "diffusion_model.transformer_blocks.1.to_q.lora_A.weight": torch.randn(rank, 4, generator=generator),
    }
    if alpha is not None:
        tensors[f"{base}.alpha"] = torch.tensor(alpha)
    save_file(tensors, str(path))
    return tensors


@pytest.mark.parametrize("keys", ["diffusers", "kohya"])
def test_the_assistant_adapter_adds_scaled_frozen_deltas_in_either_key_format(tmp_path: Path, keys: str) -> None:
    saved = _assistant_file(tmp_path / "a.safetensors", keys=keys)
    toy = Toy()
    net = QwenImage21LoRANetwork(toy, rank=1, alpha=1.0, target_modules=["nothing"])
    linear = toy.transformer_blocks[0].to_q
    x = torch.randn(3, 4)
    base = linear(x)

    attached = net.load_assistant_adapter(tmp_path / "a.safetensors", multiplier=0.5, device=CPU, dtype=torch.float32)

    prefix = "diffusion_model.transformer_blocks.0.to_q."
    down = saved[prefix + ("lora_A" if keys == "diffusers" else "lora_down") + ".weight"]
    up = saved[prefix + ("lora_B" if keys == "diffusers" else "lora_up") + ".weight"]
    assert attached == 1
    assert torch.allclose(linear(x), base + (x @ down.T @ up.T) * (8.0 / 2) * 0.5, atol=1e-5)
    layer = net.assistant_layers["transformer_blocks_0_to_q"]
    assert not any(p.requires_grad for p in layer.parameters()) and not layer.training
    assert list(net.trainable_parameters()) == []


def test_the_assistant_alpha_defaults_to_the_rank_when_the_file_has_none(tmp_path: Path) -> None:
    _assistant_file(tmp_path / "a.safetensors", rank=2, alpha=None)
    net = QwenImage21LoRANetwork(Toy(), rank=1, alpha=1.0, target_modules=["nothing"])

    net.load_assistant_adapter(tmp_path / "a.safetensors")

    assert net.assistant_layers["transformer_blocks_0_to_q"].scale == 1.0


def test_unknown_non_linear_and_half_pair_entries_are_skipped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _assistant_file(tmp_path / "a.safetensors")
    net = QwenImage21LoRANetwork(Toy(), rank=1, alpha=1.0, target_modules=["nothing"])

    with caplog.at_level(logging.WARNING):
        attached = net.load_assistant_adapter(tmp_path / "a.safetensors")

    assert attached == 1 and list(net.assistant_layers) == ["transformer_blocks_0_to_q"]
    assert "skipped 2 module(s)" in caplog.text
    assert "ghost.layer" in caplog.text and "to_out.1" in caplog.text


def test_unprefixed_assistant_keys_resolve_directly_against_the_transformer(tmp_path: Path) -> None:
    save_file(
        {
            "transformer_blocks.1.to_q.lora_A.weight": torch.ones(1, 4),
            "transformer_blocks.1.to_q.lora_B.weight": torch.ones(4, 1),
        },
        str(tmp_path / "a.safetensors"),
    )
    net = QwenImage21LoRANetwork(Toy(), rank=1, alpha=1.0, target_modules=["nothing"])

    assert net.load_assistant_adapter(tmp_path / "a.safetensors") == 1
    assert "transformer_blocks_1_to_q" in net.assistant_layers
