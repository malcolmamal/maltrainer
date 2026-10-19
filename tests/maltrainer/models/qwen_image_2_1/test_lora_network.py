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

PER_BLOCK = [
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_up",
    "img_mlp.out",
]


def _fill_up(network: QwenImage21LoRANetwork, value: float = 0.1) -> None:
    for layer in network.lora_layers.values():
        nn.init.constant_(layer.lora_up.weight, value)


def _run(model: nn.Module) -> torch.Tensor:
    latents, text, timestep = tiny_inputs()
    return model(latents, text, timestep, latent_h=2, latent_w=3)


def test_a_layer_is_an_exact_no_op_at_init_and_has_the_linears_shape() -> None:
    torch.manual_seed(0)
    layer = QwenImage21LoRALayer(nn.Linear(5, 7), rank=3, alpha=6.0)

    out = layer(torch.randn(4, 5))

    assert layer.lora_down.weight.shape == (3, 5)
    assert layer.lora_up.weight.shape == (7, 3)
    assert layer.scale == 2.0
    assert torch.equal(out, torch.zeros(4, 7))


def test_a_layer_adds_up_of_down_times_alpha_over_rank() -> None:
    layer = QwenImage21LoRALayer(nn.Linear(2, 1), rank=2, alpha=1.0)
    with torch.no_grad():
        layer.lora_down.weight.copy_(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
        layer.lora_up.weight.copy_(torch.tensor([[2.0, 3.0]]))

    out = layer(torch.tensor([[1.0, 2.0]]))

    torch.testing.assert_close(out, torch.tensor([[(2.0 * 1.0 + 3.0 * 3.0) * 0.5]]))


def test_a_layer_casts_the_input_to_its_own_dtype_outside_autocast() -> None:
    layer = QwenImage21LoRALayer(nn.Linear(2, 2), rank=1, alpha=1.0).to(torch.bfloat16)

    out = layer(torch.ones(1, 2))

    assert out.dtype == torch.bfloat16


def test_the_network_adapts_every_target_linear_in_every_block_and_nothing_else() -> None:
    network = QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=4.0)

    expected = {f"transformer_blocks.{i}.{name}" for i in range(2) for name in PER_BLOCK}
    assert set(network.name_map.values()) == expected
    assert network.name_map["transformer_blocks_1_attn_to_out_0"] == "transformer_blocks.1.attn.to_out.0"
    assert len(DEFAULT_QWEN_IMAGE_2_1_TARGET_MODULES) == 6
    assert all(isinstance(layer, QwenImage21LoRALayer) and layer.scale == 2.0 for layer in network.lora_layers.values())


@pytest.mark.parametrize(
    ("targets", "expected"),
    [(["img_mlp.out"], ["img_mlp.out"]), (["attn.to_q"], ["attn.to_q"]), (["img_in", "proj_out"], [])],
)
def test_custom_targets_replace_the_defaults_and_only_reach_into_the_blocks(
    targets: list[str], expected: list[str]
) -> None:
    network = QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=2.0, target_modules=targets)

    assert sorted(network.name_map.values()) == sorted(
        f"transformer_blocks.{i}.{n}" for i in range(2) for n in expected
    )


def test_injection_changes_nothing_until_the_up_projection_moves() -> None:
    plain = tiny_transformer(seed=2)
    expected = _run(plain)
    network = QwenImage21LoRANetwork(tiny_transformer(seed=2), rank=2, alpha=2.0)

    before = _run(network)
    _fill_up(network)
    after = _run(network)

    torch.testing.assert_close(before, expected, atol=0, rtol=0)
    assert not torch.allclose(after, expected)


def test_a_wrapped_linear_returns_its_original_output_plus_the_lora_delta() -> None:
    network = QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=4.0, target_modules=["attn.to_q"])
    linear = network.transformer.transformer_blocks[0].attn.to_q
    layer = network.lora_layers["transformer_blocks_0_attn_to_q"]
    nn.init.constant_(layer.lora_up.weight, 0.3)
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(0))

    out = linear(x)

    torch.testing.assert_close(out, x @ linear.weight.T + layer(x))


def test_a_wrapped_linear_keeps_the_output_dtype_of_the_base_layer() -> None:
    network = QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=2.0, target_modules=["attn.to_q"])
    network.transformer.transformer_blocks[0].attn.to_q.to(torch.bfloat16)

    out = network.transformer.transformer_blocks[0].attn.to_q(torch.ones(1, 16, dtype=torch.bfloat16))

    assert out.dtype == torch.bfloat16


def test_only_the_adapter_weights_are_handed_to_the_optimiser() -> None:
    network = QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=2.0)

    trainable = list(network.trainable_parameters())

    assert len(trainable) == 2 * len(network.lora_layers)
    assert {id(p) for p in trainable} == {id(p) for p in network.lora_layers.parameters()}


def test_the_adapters_alone_can_be_moved_to_another_dtype() -> None:
    network = QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=2.0)

    network.set_lora_dtype(torch.device("cpu"), torch.bfloat16)

    assert {p.dtype for p in network.lora_layers.parameters()} == {torch.bfloat16}
    assert network.transformer.img_in.weight.dtype == torch.float32


def test_the_network_forward_is_the_transformers_and_gradients_reach_only_the_adapters() -> None:
    network = QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=2.0)
    network.transformer.requires_grad_(False)
    _fill_up(network)

    _run(network).square().mean().backward()

    assert all(p.grad is not None for p in network.lora_layers.parameters())
    assert network.transformer.img_in.weight.grad is None
