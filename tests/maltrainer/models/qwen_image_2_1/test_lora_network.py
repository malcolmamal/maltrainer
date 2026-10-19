from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.qwen_image_2_1.lora import (
    DEFAULT_QWEN_IMAGE_2_1_TARGET_MODULES,
    QwenImage21LoRALayer,
    QwenImage21LoRANetwork,
)

from ._tiny import tiny_inputs, tiny_transformer


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)
        self.other = nn.Linear(4, 4, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(4, 4, bias=False), nn.Identity()])


class Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block(), Block()])
        self.to_q = nn.Linear(4, 4, bias=False)  # outside the blocks: must never be adapted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer_blocks:
            x = block.to_out[0](block.to_q(x)) + block.other(x)
        return x


def test_a_fresh_layer_scales_alpha_over_rank_and_adds_nothing() -> None:
    layer = QwenImage21LoRALayer(nn.Linear(6, 3), rank=2, alpha=6.0)

    assert layer.scale == 3.0
    assert layer.lora_down.weight.shape == (2, 6) and layer.lora_up.weight.shape == (3, 2)
    assert torch.equal(layer(torch.randn(5, 6)), torch.zeros(5, 3))


def test_the_layer_output_is_up_of_down_times_scale() -> None:
    layer = QwenImage21LoRALayer(nn.Linear(3, 2), rank=1, alpha=2.0)
    layer.lora_down.weight.data = torch.tensor([[1.0, 2.0, 3.0]])
    layer.lora_up.weight.data = torch.tensor([[0.5], [-1.0]])

    out = layer(torch.tensor([[1.0, 1.0, 1.0]]))

    assert torch.allclose(out, torch.tensor([[6.0 * 0.5 * 2.0, -6.0 * 2.0]]))


def test_the_layer_casts_float_input_to_its_own_dtype_but_leaves_integer_input_alone() -> None:
    layer = QwenImage21LoRALayer(nn.Linear(3, 2), rank=1, alpha=1.0).to(torch.bfloat16)

    assert layer(torch.randn(2, 3, dtype=torch.float32)).dtype == torch.bfloat16
    with pytest.raises(RuntimeError):
        layer(torch.ones(2, 3, dtype=torch.int64))


def test_only_target_linears_inside_transformer_blocks_are_adapted() -> None:
    net = QwenImage21LoRANetwork(Toy(), rank=2, alpha=2.0)

    assert sorted(net.name_map.values()) == [
        "transformer_blocks.0.to_out.0",
        "transformer_blocks.0.to_q",
        "transformer_blocks.1.to_out.0",
        "transformer_blocks.1.to_q",
    ]
    assert net.name_map["transformer_blocks_0_to_out_0"] == "transformer_blocks.0.to_out.0"


def test_custom_target_modules_replace_the_default_list() -> None:
    net = QwenImage21LoRANetwork(Toy(), rank=2, alpha=2.0, target_modules=["other"])

    assert sorted(net.name_map.values()) == ["transformer_blocks.0.other", "transformer_blocks.1.other"]


def test_a_freshly_injected_network_leaves_the_transformer_output_unchanged() -> None:
    toy = Toy()
    x = torch.randn(3, 4)
    before = toy(x)

    QwenImage21LoRANetwork(toy, rank=2, alpha=2.0)

    assert torch.allclose(toy(x), before)


def test_the_injected_delta_is_added_to_the_base_output_of_the_wrapped_linear() -> None:
    toy = Toy()
    net = QwenImage21LoRANetwork(toy, rank=1, alpha=1.0, target_modules=["to_q"])
    layer = net.lora_layers["transformer_blocks_0_to_q"]
    layer.lora_down.weight.data = torch.ones(1, 4)
    layer.lora_up.weight.data = torch.full((4, 1), 2.0)
    linear = toy.transformer_blocks[0].to_q
    x = torch.randn(3, 4)

    expected = x @ linear.weight.T + (x.sum(-1, keepdim=True) * 2.0)

    assert torch.allclose(linear(x), expected, atol=1e-6)


def test_a_delta_in_another_dtype_is_cast_back_to_the_base_dtype() -> None:
    toy = Toy()
    net = QwenImage21LoRANetwork(toy, rank=1, alpha=1.0, target_modules=["to_q"])
    net.set_lora_dtype(torch.device("cpu"), torch.float64)
    net.lora_layers["transformer_blocks_0_to_q"].lora_up.weight.data.fill_(1.0)

    out = toy.transformer_blocks[0].to_q(torch.randn(2, 4))

    assert out.dtype == torch.float32


def test_trainable_parameters_are_only_the_adapter_weights() -> None:
    net = QwenImage21LoRANetwork(Toy(), rank=2, alpha=2.0)

    params = list(net.trainable_parameters())

    assert len(params) == 2 * 4
    assert {id(p) for p in params} == {id(p) for p in net.lora_layers.parameters()}


def test_the_default_targets_cover_every_projection_of_the_real_transformer() -> None:
    transformer = tiny_transformer()
    net = QwenImage21LoRANetwork(transformer, rank=2, alpha=2.0)

    block0 = {n for n in net.name_map.values() if n.startswith("transformer_blocks.0.")}

    assert len(DEFAULT_QWEN_IMAGE_2_1_TARGET_MODULES) == 6
    assert any(n.endswith("to_out.0") for n in block0) and any(n.endswith("to_q") for n in block0)
    assert any(n.endswith("img_mlp.gate_up") for n in net.name_map.values())
    assert any(n.endswith("img_mlp.out") for n in net.name_map.values())


def test_forward_delegates_to_the_wrapped_transformer() -> None:
    transformer = tiny_transformer()
    net = QwenImage21LoRANetwork(transformer, rank=2, alpha=2.0)
    latents, text, timestep = tiny_inputs()

    assert torch.equal(net(latents, text, timestep, 2, 3), transformer(latents, text, timestep, 2, 3))
