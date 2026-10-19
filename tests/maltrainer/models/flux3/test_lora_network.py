from __future__ import annotations

import logging

import pytest
import torch
from torch import nn

from maltrainer.models.flux3.lora import Flux3LoRANetwork
from maltrainer.models.flux3.model import Flux3Config
from tests.maltrainer.models.flux3.tiny_model import TINY, tiny_inputs, tiny_transformer

FUSED = Flux3Config(hidden_size=128, num_heads=1, depth=1, depth_single_blocks=2, fused_qkv=True)


@pytest.mark.parametrize(
    ("config", "targets", "count"),
    [(TINY, "blocks", 24), (TINY, "single", 12), (TINY, "all", 34), (FUSED, "blocks", 24), (FUSED, "single", 12)],
)
def test_each_preset_adapts_the_expected_number_of_linears(config: Flux3Config, targets: str, count: int) -> None:
    network = Flux3LoRANetwork(tiny_transformer(config), rank=2, alpha=2.0, targets=targets)

    assert len(network.lora_layers) == len(network.name_map) == count


def test_the_adapted_names_of_the_single_preset_are_the_single_block_projections() -> None:
    network = Flux3LoRANetwork(tiny_transformer(), rank=2, targets="single")

    assert {"single_blocks.0.q_proj", "single_blocks.1.mlp_out"} <= set(network.name_map.values())
    assert all(name.startswith("single_blocks.") for name in network.name_map.values())
    assert network.name_map["single_blocks_0_q_proj"] == "single_blocks.0.q_proj"


def test_a_fused_qkv_gets_three_separate_adapters_that_each_cover_a_third_of_it() -> None:
    network = Flux3LoRANetwork(tiny_transformer(FUSED), rank=2, targets="single")

    names = set(network.name_map.values())
    assert {"single_blocks.0.q_proj", "single_blocks.0.k_proj", "single_blocks.0.v_proj"} <= names
    assert not any(name.endswith("qkv") for name in names)
    assert network.lora_layers["single_blocks_0_k_proj"].lora_up.weight.shape == (128, 2)


@pytest.mark.parametrize("config", [TINY, FUSED])
def test_a_fresh_network_leaves_the_model_output_unchanged(config: Flux3Config) -> None:
    model = tiny_transformer(config)
    inputs = tiny_inputs()
    before = model(*inputs)

    network = Flux3LoRANetwork(model, rank=4, alpha=8.0, targets="all")

    assert torch.equal(network(*inputs), before)


def test_a_trained_adapter_shifts_the_output_and_receives_a_gradient() -> None:
    model = tiny_transformer()
    inputs = tiny_inputs()
    before = model(*inputs)
    network = Flux3LoRANetwork(model, rank=2, alpha=2.0, targets="single")
    network.lora_layers["single_blocks_0_q_proj"].lora_up.weight.data.normal_(std=0.5)

    out = network(*inputs)
    out.square().sum().backward()

    assert not torch.allclose(out, before)
    assert network.lora_layers["single_blocks_0_q_proj"].lora_up.weight.grad is not None


def test_only_the_adapter_weights_are_trainable_parameters() -> None:
    network = Flux3LoRANetwork(tiny_transformer(), rank=2, targets="blocks")

    trainable = list(network.trainable_parameters())

    assert len(trainable) == 2 * len(network.lora_layers)
    assert all(p.requires_grad for p in trainable)
    assert {id(p) for p in trainable}.isdisjoint({id(p) for p in network.transformer.parameters()})


def test_the_adapters_can_be_moved_to_their_own_dtype() -> None:
    network = Flux3LoRANetwork(tiny_transformer(), rank=2, targets="single")

    network.set_lora_dtype(torch.device("cpu"), torch.bfloat16)

    assert {p.dtype for p in network.trainable_parameters()} == {torch.bfloat16}
    assert {p.dtype for p in network.transformer.parameters()} == {torch.float32}


def test_one_param_group_holds_every_adapter_unless_splitting_by_family() -> None:
    network = Flux3LoRANetwork(tiny_transformer(), rank=2, targets="all")

    (single,) = network.param_groups(split=False)
    split = {group["name"]: group["params"] for group in network.param_groups(split=True)}

    assert single["name"] == "lora" and len(single["params"]) == 2 * 34
    assert {name: len(params) for name, params in split.items()} == {"attn": 2 * 16, "mlp": 2 * 8, "other": 2 * 10}


def test_the_state_dict_holds_comfyuis_lora_a_lora_b_and_alpha_under_the_prefix() -> None:
    network = Flux3LoRANetwork(tiny_transformer(), rank=2, alpha=6.0, targets="single")
    layer = network.lora_layers["single_blocks_1_mlp_out"]

    state = network.get_state_dict()

    assert len(state) == 3 * 12
    key = "diffusion_model.single_blocks.1.mlp_out"
    assert torch.equal(state[f"{key}.lora_A.weight"], layer.lora_down.weight.detach().to(torch.bfloat16))
    assert torch.equal(state[f"{key}.lora_B.weight"], layer.lora_up.weight.detach().to(torch.bfloat16))
    assert state[f"{key}.alpha"].dtype == torch.float32 and state[f"{key}.alpha"].item() == 6.0


def test_the_state_dict_keeps_the_training_dtype_when_no_save_dtype_is_given() -> None:
    network = Flux3LoRANetwork(tiny_transformer(), rank=2, targets="single", prefix="")

    state = network.get_state_dict(save_dtype=None)

    assert "single_blocks.0.q_proj.lora_A.weight" in state
    assert state["single_blocks.0.q_proj.lora_B.weight"].dtype == torch.float32


def test_the_network_logs_how_many_adapters_each_family_got(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="maltrainer.models.flux3.lora"):
        Flux3LoRANetwork(tiny_transformer(), rank=2, alpha=4.0, targets="single")

    assert "12 adapters ({'attn': 8, 'mlp': 4}), rank 2, alpha 4.0, targets single" in caplog.text


def test_an_unknown_target_preset_is_refused() -> None:
    with pytest.raises(ValueError, match="lora.targets must be one of"):
        Flux3LoRANetwork(tiny_transformer(), targets="attn")


def test_a_model_with_nothing_to_adapt_is_refused() -> None:
    with pytest.raises(ValueError, match="No FLUX.3 modules matched"):
        Flux3LoRANetwork(nn.Sequential(nn.Linear(2, 2)))
