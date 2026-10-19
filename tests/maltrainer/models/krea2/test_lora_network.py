from __future__ import annotations

import logging

import pytest
import torch

from maltrainer.models.comfy_fp8 import swap_in_fp8_linears
from maltrainer.models.krea2.lora import Krea2LoRANetwork
from tests.maltrainer.models.krea2.tiny_model import tiny_inputs, tiny_transformer


def test_a_fresh_network_leaves_the_model_output_unchanged() -> None:
    model = tiny_transformer()
    inputs = tiny_inputs()
    before = model(*inputs)

    network = Krea2LoRANetwork(model, rank=4, alpha=8.0, targets="all")

    assert torch.equal(network(*inputs), before)


def test_a_wrapped_linear_adds_exactly_up_of_down_of_its_input_times_alpha_over_rank() -> None:
    model = tiny_transformer()
    network = Krea2LoRANetwork(model, rank=2, alpha=6.0)
    layer = network.lora_layers["blocks_0_mlp_gate"]
    layer.lora_up.weight.data.normal_()
    x = torch.randn(3, 32)
    linear = model.blocks[0].mlp.gate

    delta = linear(x) - torch.nn.functional.linear(x, linear.weight)

    expected = layer.lora_up(layer.lora_down(x)) * 3.0
    torch.testing.assert_close(delta, expected, atol=1e-6, rtol=1e-5)


def test_disabled_runs_the_base_model_alone_and_restores_the_adapters_afterwards() -> None:
    model = tiny_transformer()
    inputs = tiny_inputs()
    base = model(*inputs)
    network = Krea2LoRANetwork(model, rank=2)
    for layer in network.lora_layers.values():
        layer.lora_up.weight.data.normal_(std=0.3)
    adapted = network(*inputs)

    with network.disabled():
        inside = network(*inputs)
    after = network(*inputs)

    assert torch.equal(inside, base)
    assert not torch.allclose(adapted, base)
    assert torch.equal(after, adapted)


def test_disabled_restores_the_adapters_even_when_the_block_raises() -> None:
    network = Krea2LoRANetwork(tiny_transformer(), rank=2)

    with pytest.raises(RuntimeError, match="boom"), network.disabled():
        assert network._active == [False]
        raise RuntimeError("boom")

    assert network._active == [True]


def test_a_float32_adapter_on_an_fp8_base_returns_the_base_dtype() -> None:
    model = tiny_transformer().to(torch.bfloat16)
    model.compute_dtype = torch.bfloat16
    layers = {f"blocks.{i}.mlp.gate": {"format": "float8_e4m3fn"} for i in range(2)}
    swap_in_fp8_linears(model, layers, torch.bfloat16)
    network = Krea2LoRANetwork(model, rank=2, block_range=(0, 1))
    module = model.blocks[0].mlp.gate
    module.weight = module.weight.float().to(torch.float8_e4m3fn)

    out = module(torch.randn(2, 32, dtype=torch.bfloat16))

    assert out.dtype == torch.bfloat16
    assert network.lora_layers["blocks_0_mlp_gate"].lora_down.weight.dtype == torch.float32


def test_only_the_adapter_weights_are_trainable_parameters() -> None:
    network = Krea2LoRANetwork(tiny_transformer(), rank=2)

    trainable = list(network.trainable_parameters())

    assert len(trainable) == 2 * len(network.lora_layers)
    assert {id(p) for p in trainable}.isdisjoint({id(p) for p in network.transformer.parameters()})


def test_the_adapters_can_be_moved_to_their_own_dtype() -> None:
    network = Krea2LoRANetwork(tiny_transformer(), rank=2)

    network.set_lora_dtype(torch.device("cpu"), torch.bfloat16)

    assert {p.dtype for p in network.trainable_parameters()} == {torch.bfloat16}
    assert {p.dtype for p in network.transformer.parameters()} == {torch.float32}


def test_one_param_group_holds_every_adapter_unless_splitting_by_family() -> None:
    network = Krea2LoRANetwork(tiny_transformer(), rank=2, targets="all")

    (single,) = network.param_groups(split=False)
    split = {group["name"]: len(group["params"]) for group in network.param_groups(split=True)}

    assert single["name"] == "lora" and len(single["params"]) == 2 * 56
    assert split == {"attn": 2 * 10, "mlp": 2 * 6, "txtfusion": 2 * 33, "other": 2 * 7}


def test_the_state_dict_holds_comfyuis_lora_a_lora_b_and_alpha_under_the_prefix() -> None:
    network = Krea2LoRANetwork(tiny_transformer(), rank=2, alpha=6.0)
    layer = network.lora_layers["blocks_1_attn_wo"]

    state = network.get_state_dict()

    key = "diffusion_model.blocks.1.attn.wo"
    assert len(state) == 3 * 16
    assert torch.equal(state[f"{key}.lora_A.weight"], layer.lora_down.weight.detach().to(torch.bfloat16))
    assert state[f"{key}.lora_B.weight"].dtype == torch.bfloat16
    assert state[f"{key}.alpha"].dtype == torch.float32 and state[f"{key}.alpha"].item() == 6.0


def test_the_state_dict_keeps_the_training_dtype_when_no_save_dtype_is_given() -> None:
    network = Krea2LoRANetwork(tiny_transformer(), rank=2, prefix="")

    state = network.get_state_dict(save_dtype=None)

    assert state["blocks.0.attn.wq.lora_B.weight"].dtype == torch.float32


def test_the_network_logs_how_many_adapters_each_family_got(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="maltrainer.models.krea2.lora"):
        Krea2LoRANetwork(tiny_transformer(), rank=2, alpha=4.0)

    assert "16 layers ({'attn': 10, 'mlp': 6}), rank 2, alpha 4.0, targets attn-mlp" in caplog.text
