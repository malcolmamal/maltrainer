from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRANetwork
from maltrainer.models.minimax_h3.model import MiniMaxH3TransformerOutput
from tests.maltrainer.models.minimax_h3.h3_tiny import Packed, tiny_transformer


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.qkv_proj.weight.copy_(torch.tensor([[1.0, 1.0]]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qkv_proj(x)


def _set_hand_weights(network: MiniMaxH3LoRANetwork, up: float = 1.0) -> None:
    layer = network.lora_layers["qkv_proj"]
    with torch.no_grad():
        layer.lora_down.weight.copy_(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
        layer.lora_up.weight.copy_(torch.tensor([[2.0 * up, 3.0 * up]]))


def test_a_fresh_network_leaves_the_transformer_output_bit_identical() -> None:
    transformer = tiny_transformer()
    inputs = Packed(text=2, audio=1, video=3).inputs()
    before = transformer(**inputs)

    network = MiniMaxH3LoRANetwork(transformer, rank=2, alpha=4.0)
    after = network(**inputs)

    assert isinstance(after, MiniMaxH3TransformerOutput)
    assert torch.equal(after.video, before.video)
    assert torch.equal(after.audio, before.audio)


def test_a_wrapped_linear_adds_up_of_down_times_alpha_over_rank_to_its_output() -> None:
    network = MiniMaxH3LoRANetwork(_Toy(), rank=2, alpha=4.0)
    _set_hand_weights(network)

    out = network(torch.tensor([[1.0, 2.0]]))

    assert out.item() == 3.0 + (2.0 * 1.0 + 3.0 * 3.0) * 2.0


def test_every_wrapped_linear_of_the_transformer_returns_base_plus_delta() -> None:
    network = MiniMaxH3LoRANetwork(tiny_transformer(), rank=2, alpha=4.0)
    layer = network.lora_layers["blocks_1_mlp_fc1"]
    torch.nn.init.normal_(layer.lora_up.weight)
    module = network.transformer.blocks[1].mlp.fc1
    x = torch.randn(3, 16)

    out = module(x)

    assert torch.equal(out, F.linear(x, module.weight) + layer(x))
    assert not torch.equal(out, F.linear(x, module.weight))


def test_an_adapter_held_in_higher_precision_is_summed_before_rounding_to_the_base_dtype() -> None:
    network = MiniMaxH3LoRANetwork(_Toy(), rank=2, alpha=4.0)
    network.set_lora_dtype(torch.device("cpu"), torch.float64)
    _set_hand_weights(network, up=1e-3)
    x = torch.tensor([[1.0, 2.0]])

    out = network(x)

    delta = network.lora_layers["qkv_proj"](x)
    assert out.dtype == torch.float32
    assert delta.dtype == torch.float64
    assert out.item() == (torch.tensor(3.0, dtype=torch.float64) + delta.squeeze()).float().item()


def test_two_networks_on_one_transformer_add_their_deltas_whatever_the_order() -> None:
    toy = _Toy()
    first = MiniMaxH3LoRANetwork(toy, rank=2, alpha=4.0)
    second = MiniMaxH3LoRANetwork(toy, rank=2, alpha=2.0)
    _set_hand_weights(first)
    _set_hand_weights(second, up=0.5)

    out = toy(torch.tensor([[1.0, 2.0]]))

    assert out.item() == 3.0 + 22.0 + 11.0 * 0.5


def test_set_lora_dtype_casts_only_the_adapters() -> None:
    network = MiniMaxH3LoRANetwork(tiny_transformer(), rank=2, alpha=4.0)

    network.set_lora_dtype(torch.device("cpu"), torch.bfloat16)

    assert {p.dtype for p in network.lora_layers.parameters()} == {torch.bfloat16}
    assert {p.dtype for p in network.transformer.parameters()} == {torch.float32}


def test_trainable_parameters_are_the_adapter_matrices_and_nothing_of_the_frozen_base() -> None:
    network = MiniMaxH3LoRANetwork(tiny_transformer(), rank=2, alpha=4.0)

    trainable = list(network.trainable_parameters())

    expected = [w for layer in network.lora_layers.values() for w in (layer.lora_down.weight, layer.lora_up.weight)]
    assert len(trainable) == 2 * len(network.lora_layers)
    assert {id(p) for p in trainable} == {id(p) for p in expected}


def test_the_backward_pass_reaches_the_adapters_and_not_the_frozen_base() -> None:
    transformer = tiny_transformer().requires_grad_(False)
    network = MiniMaxH3LoRANetwork(transformer, rank=2, alpha=4.0)
    out = network(**Packed(text=2, audio=1, video=3).inputs())

    (out.video.sum() + out.audio.sum()).backward()

    up = network.lora_layers["blocks_0_attn_out_proj"].lora_up.weight
    assert up.grad is not None
    assert up.grad.abs().sum() > 0
    assert all(p.grad is None for p in transformer.parameters())


def test_the_state_dict_is_a_prefixed_a_b_alpha_triple_per_layer_stored_in_bfloat16() -> None:
    network = MiniMaxH3LoRANetwork(tiny_transformer(), rank=2, alpha=4.0)
    layer = network.lora_layers["blocks_2_mlp_fc2"]
    torch.nn.init.normal_(layer.lora_up.weight)

    state = network.get_state_dict()

    key = "diffusion_model.blocks.2.mlp.fc2"
    assert len(state) == 3 * len(network.lora_layers)
    assert state[f"{key}.lora_A.weight"].dtype == torch.bfloat16
    torch.testing.assert_close(state[f"{key}.lora_B.weight"], layer.lora_up.weight.detach().bfloat16())
    assert state[f"{key}.alpha"].dtype == torch.float32
    assert state[f"{key}.alpha"].item() == 4.0


def test_the_state_dict_can_keep_the_training_dtype_and_use_another_prefix() -> None:
    network = MiniMaxH3LoRANetwork(tiny_transformer(), rank=2, alpha=4.0, prefix="lora.")

    state = network.get_state_dict(save_dtype=None)

    weights = [value for key, value in state.items() if key.endswith("weight")]
    assert all(key.startswith("lora.") for key in state)
    assert {value.dtype for value in weights} == {torch.float32}
    assert not any(value.requires_grad for value in state.values())
