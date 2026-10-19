from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.lora import LoRALayer, LoRANetwork


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(4, 4)
        self.to_k = nn.Linear(4, 4)
        self.to_v = nn.Linear(4, 4)
        self.to_out = nn.ModuleList([nn.Linear(4, 4), nn.Dropout(0.0)])
        self.ff = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.to_q(x) + self.to_k(x) + self.to_v(x)
        return self.ff(self.to_out[1](self.to_out[0](mixed)))


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Attention()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x)


ADAPTED = ["blocks.0.to_q", "blocks.0.to_k", "blocks.0.to_v", "blocks.0.to_out.0"]


def _network(**options: object) -> LoRANetwork:
    torch.manual_seed(0)
    return LoRANetwork(_Toy(), rank=2, alpha=4.0, **options)


def test_a_layer_has_down_and_up_projections_and_starts_as_a_no_op() -> None:
    torch.manual_seed(0)

    layer = LoRALayer(nn.Linear(3, 5), rank=2, alpha=1.0)

    assert layer.lora_down.weight.shape == (2, 3)
    assert layer.lora_up.weight.shape == (5, 2)
    torch.testing.assert_close(layer(torch.randn(4, 3)), torch.zeros(4, 5))


def test_a_layer_output_is_up_of_down_times_alpha_over_rank() -> None:
    layer = LoRALayer(nn.Linear(2, 1), rank=2, alpha=1.0)
    with torch.no_grad():
        layer.lora_down.weight.copy_(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
        layer.lora_up.weight.copy_(torch.tensor([[2.0, 3.0]]))

    plain = layer(torch.tensor([[1.0, 2.0]]))
    layer.set_alpha(4)

    torch.testing.assert_close(plain, torch.tensor([[(2.0 * 1.0 + 3.0 * 3.0) * 0.5]]))
    torch.testing.assert_close(layer(torch.tensor([[1.0, 2.0]])), torch.tensor([[11.0 * 2.0]]))
    assert (layer.alpha, layer.scale) == (4.0, 2.0)


def test_a_layer_rejects_modules_other_than_linear() -> None:
    with pytest.raises(ValueError, match="Unsupported module type"):
        LoRALayer(nn.Conv2d(1, 1, 1), rank=1, alpha=1.0)


def test_the_network_adapts_only_linears_whose_name_matches_a_target() -> None:
    network = _network()

    assert sorted(network.name_map.values()) == sorted(ADAPTED)
    assert network.name_map["blocks_0_to_out_0"] == "blocks.0.to_out.0"
    assert all(layer.rank == 2 and layer.scale == 2.0 for layer in network.lora_layers.values())


def test_custom_targets_replace_the_default_attention_projections() -> None:
    network = _network(target_modules=["ff"])

    assert list(network.name_map.values()) == ["blocks.0.ff"]


def test_the_network_output_adds_the_lora_delta_to_the_original_output() -> None:
    network = _network()
    x = torch.randn(3, 4)
    base = network(x).detach()
    layer = network.lora_layers["blocks_0_to_q"]
    with torch.no_grad():
        layer.lora_up.weight.fill_(0.5)

    adapted = network(x)

    block = network.transformer.blocks[0]
    expected = layer(x) @ block.to_out[0].weight.T @ block.ff.weight.T
    torch.testing.assert_close(adapted - base, expected.detach())


def test_the_state_dict_uses_prefixed_dot_names_with_a_and_b_matrices() -> None:
    network = _network()
    network.prefix = "diffusion_model."

    state = network.get_state_dict()

    assert len(state) == 3 * len(ADAPTED)
    layer = network.lora_layers["blocks_0_to_v"]
    assert state["diffusion_model.blocks.0.to_v.lora_A.weight"] is layer.lora_down.weight
    assert state["diffusion_model.blocks.0.to_v.lora_B.weight"] is layer.lora_up.weight
    assert state["diffusion_model.blocks.0.to_v.alpha"].item() == 4.0


@pytest.mark.parametrize("name", ["blocks.0.to_k", "blocks_0_to_k"])
def test_weights_and_alpha_load_by_dot_name_or_safe_name(name: str) -> None:
    network = _network()
    down, up = torch.full((2, 4), 0.25), torch.full((4, 2), -0.5)

    network.load_state_dict({f"{name}.lora_down.weight": down, f"{name}.lora_up.weight": up, f"{name}.alpha": 1.0})

    layer = network.lora_layers["blocks_0_to_k"]
    torch.testing.assert_close(layer.lora_down.weight.detach(), down)
    torch.testing.assert_close(layer.lora_up.weight.detach(), up)
    assert layer.scale == 0.5


def test_weights_without_an_alpha_keep_the_configured_scale() -> None:
    network = _network()

    network.load_state_dict(
        {"blocks.0.to_q.lora_down.weight": torch.ones(2, 4), "blocks.0.to_q.lora_up.weight": torch.ones(4, 2)}
    )

    assert network.lora_layers["blocks_0_to_q"].scale == 2.0


@pytest.mark.parametrize("key", ["diffusion_model.blocks.0.to_v", "lora_unet_blocks_0_to_v"])
def test_weights_under_an_unknown_prefix_are_found_by_suffix(key: str) -> None:
    network = _network()
    untouched = network.lora_layers["blocks_0_to_q"].lora_down.weight.detach().clone()
    down, up = torch.full((2, 4), 0.125), torch.full((4, 2), 2.0)

    network.load_state_dict({f"{key}.lora_down.weight": down, f"{key}.lora_up.weight": up})

    layer = network.lora_layers["blocks_0_to_v"]
    torch.testing.assert_close(layer.lora_down.weight.detach(), down)
    torch.testing.assert_close(layer.lora_up.weight.detach(), up)
    torch.testing.assert_close(network.lora_layers["blocks_0_to_q"].lora_down.weight.detach(), untouched)
