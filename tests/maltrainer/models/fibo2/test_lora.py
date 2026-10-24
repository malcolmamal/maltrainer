from __future__ import annotations

import pytest
import torch

from maltrainer.models.fibo2.lora import Fibo2LoRANetwork, param_group_of, selected

from .tiny_model import tiny_inputs, tiny_transformer


@pytest.mark.parametrize(("targets", "count"), [("blocks", 43), ("main", 39), ("all", 63)])
def test_each_preset_adapts_the_expected_number_of_linears(targets: str, count: int) -> None:
    network = Fibo2LoRANetwork(tiny_transformer(), rank=2, alpha=2.0, targets=targets)

    assert len(network.lora_layers) == len(network.name_map) == count


def test_the_fused_projections_get_one_adapter_each_under_the_file_s_module_names() -> None:
    network = Fibo2LoRANetwork(tiny_transformer(), rank=2)

    names = set(network.name_map.values())
    assert {"layers.0.attention.qkv", "layers.0.feed_forward.w13", "layers.1.cross_attention.to_kv"} <= names
    assert {"noise_refiner.0.attention.out", "layers.5.feed_forward.w2"} <= names
    assert network.lora_layers["layers_0_attention_qkv"].lora_up.weight.shape == (3 * 256, 2)
    assert network.name_map["layers_0_attention_qkv"] == "layers.0.attention.qkv"


def test_the_main_preset_leaves_the_refiner_alone_and_blocks_leaves_the_perceiver_and_io_alone() -> None:
    main = Fibo2LoRANetwork(tiny_transformer(), rank=2, targets="main")
    blocks = Fibo2LoRANetwork(tiny_transformer(), rank=2, targets="blocks")

    assert not any(name.startswith("noise_refiner.") for name in main.name_map.values())
    assert not any(
        name.startswith(("perceiver.", "x_embedder", "final_layer", "t_embedder")) for name in blocks.name_map.values()
    )


def test_the_all_preset_adds_the_perceiver_the_modulations_and_the_io_layers() -> None:
    names = set(Fibo2LoRANetwork(tiny_transformer(), rank=2, targets="all").name_map.values())

    assert {
        "perceiver.layers.0.cross_attention.to_kv",
        "perceiver.layers.0.self_attention.to_q",
        "perceiver.layers.0.ffn.w13",
        "layers.0.adaLN_modulation.0",
        "noise_refiner.0.adaLN_modulation.0",
        "x_embedder",
        "t_embedder.mlp.0",
        "t_embedder.mlp.2",
        "final_layer.linear",
        "final_layer.adaLN_modulation.1",
    } <= names


@pytest.mark.parametrize("targets", ["blocks", "main", "all"])
def test_a_fresh_network_leaves_the_model_output_unchanged(targets: str) -> None:
    model = tiny_transformer()
    inputs = tiny_inputs()
    before = model(*inputs)

    network = Fibo2LoRANetwork(model, rank=4, alpha=8.0, targets=targets)

    assert torch.equal(network(*inputs), before)


def test_a_trained_adapter_shifts_the_output_and_receives_a_gradient() -> None:
    model = tiny_transformer().requires_grad_(False)
    inputs = tiny_inputs()
    before = model(*inputs)
    network = Fibo2LoRANetwork(model, rank=2, alpha=2.0)
    network.lora_layers["layers_0_attention_qkv"].lora_up.weight.data.normal_(std=0.5)

    after = network(*inputs)
    after.square().sum().backward()

    assert not torch.allclose(after, before)
    assert network.lora_layers["layers_0_attention_qkv"].lora_down.weight.grad is not None
    assert all(p.grad is None for p in model.parameters())


def test_the_state_dict_holds_comfyui_keys_with_the_alpha() -> None:
    network = Fibo2LoRANetwork(tiny_transformer(), rank=2, alpha=4.0)

    state = network.get_state_dict(torch.float16)

    assert state["diffusion_model.layers.0.attention.qkv.lora_A.weight"].shape == (2, 256)
    assert state["diffusion_model.layers.0.attention.qkv.lora_B.weight"].shape == (768, 2)
    assert state["diffusion_model.layers.0.attention.qkv.lora_B.weight"].dtype == torch.float16
    assert state["diffusion_model.layers.0.attention.qkv.alpha"].item() == 4.0
    assert len(state) == 3 * len(network.lora_layers)


def test_the_prefix_and_the_unconverted_dtype_are_honoured() -> None:
    network = Fibo2LoRANetwork(tiny_transformer(), rank=2, prefix="transformer.")

    state = network.get_state_dict(None)

    assert "transformer.layers.0.attention.qkv.lora_A.weight" in state
    assert state["transformer.layers.0.attention.qkv.lora_A.weight"].dtype == torch.float32


def test_the_optimizer_groups_split_into_attention_mlp_and_other() -> None:
    network = Fibo2LoRANetwork(tiny_transformer(), rank=2, targets="all")

    groups = {g["name"]: g["params"] for g in network.param_groups(split=True)}
    whole = network.param_groups(split=False)

    assert set(groups) == {"attn", "mlp", "other"}
    assert sum(len(p) for p in groups.values()) == len(whole[0]["params"]) == 2 * len(network.lora_layers)
    assert whole[0]["name"] == "lora"


def test_the_lora_dtype_is_applied_to_the_adapters_only() -> None:
    model = tiny_transformer()
    network = Fibo2LoRANetwork(model, rank=2)

    network.set_lora_dtype("cpu", torch.bfloat16)

    assert {p.dtype for p in network.trainable_parameters()} == {torch.bfloat16}
    assert {p.dtype for p in model.parameters()} == {torch.float32}


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("layers.0.attention.qkv", "attn"),
        ("layers.1.cross_attention.to_out", "attn"),
        ("layers.0.feed_forward.w2", "mlp"),
        ("x_embedder", "other"),
    ],
)
def test_modules_group_by_family(name: str, group: str) -> None:
    assert param_group_of(name) == group


def test_an_unknown_preset_is_refused() -> None:
    with pytest.raises(ValueError, match=r"lora\.targets must be one of"):
        Fibo2LoRANetwork(tiny_transformer(), targets="everything")
    with pytest.raises(ValueError, match=r"lora\.targets must be one of"):
        selected("layers.0.attention.qkv", "everything")


def test_a_model_without_matching_linears_is_refused() -> None:
    with pytest.raises(ValueError, match=r"No Fibo 2\.0 modules matched"):
        Fibo2LoRANetwork(torch.nn.Sequential(torch.nn.Linear(2, 2)), rank=2)
