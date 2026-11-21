from __future__ import annotations

import pytest
import torch

from maltrainer.models.visage.lora import VisageLoRANetwork, param_group_of, selected

from .tiny_model import tiny_inputs, tiny_transformer


@pytest.mark.parametrize(("targets", "count"), [("blocks", 8), ("attn", 4), ("all", 15)])
def test_each_preset_adapts_the_expected_number_of_linears(targets: str, count: int) -> None:
    network = VisageLoRANetwork(tiny_transformer(), rank=2, alpha=2.0, targets=targets)

    assert len(network.lora_layers) == len(network.name_map) == count


def test_the_blocks_get_their_fused_attention_and_swiglu_linears_under_the_module_names_of_the_file() -> None:
    network = VisageLoRANetwork(tiny_transformer(), rank=2)

    assert set(network.name_map.values()) == {
        f"blocks.{i}.{leaf}" for i in range(2) for leaf in ("attn.qkv", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    }
    assert network.lora_layers["blocks_0_attn_qkv"].lora_up.weight.shape == (3 * 128, 2)
    assert network.lora_layers["blocks_1_mlp_fc1"].lora_up.weight.shape == (2 * 2 * 128, 2)


def test_the_attn_preset_leaves_the_swiglu_alone_and_blocks_leaves_the_modulations_and_io_alone() -> None:
    attn = set(VisageLoRANetwork(tiny_transformer(), rank=2, targets="attn").name_map.values())
    blocks = set(VisageLoRANetwork(tiny_transformer(), rank=2).name_map.values())

    assert not any(".mlp." in name for name in attn)
    io = ("text_proj", "time_embed", "final_mod", "out_proj")
    assert not any(name.startswith(io) or ".mod." in name for name in blocks)


def test_the_all_preset_adds_the_modulations_and_the_text_timestep_and_output_linears() -> None:
    names = set(VisageLoRANetwork(tiny_transformer(), rank=2, targets="all").name_map.values())

    assert {"blocks.0.mod.1", "blocks.1.mod.1", "text_proj", "time_embed.1", "time_embed.3"} <= names
    assert {"out_proj", "final_mod.1"} <= names


@pytest.mark.parametrize("targets", ["blocks", "attn", "all"])
def test_a_fresh_network_leaves_the_model_output_unchanged(targets: str) -> None:
    model = tiny_transformer()
    inputs = tiny_inputs()
    before = model(*inputs)

    network = VisageLoRANetwork(model, rank=4, alpha=8.0, targets=targets)

    assert torch.equal(network(*inputs), before)


def test_a_trained_adapter_shifts_the_output_and_receives_a_gradient() -> None:
    model = tiny_transformer().requires_grad_(False)
    inputs = tiny_inputs()
    before = model(*inputs)
    network = VisageLoRANetwork(model, rank=2, alpha=2.0)
    network.lora_layers["blocks_0_attn_qkv"].lora_up.weight.data.normal_(std=0.5)

    after = network(*inputs)
    after.square().sum().backward()

    assert not torch.allclose(after, before)
    assert network.lora_layers["blocks_0_attn_qkv"].lora_down.weight.grad is not None
    assert all(p.grad is None for p in model.parameters())


def test_a_float32_adapter_on_a_bf16_model_returns_the_model_s_dtype() -> None:
    model = tiny_transformer().bfloat16()
    network = VisageLoRANetwork(model, rank=2)
    x, t, text = tiny_inputs()

    out = network(x.bfloat16(), t, text.bfloat16())

    assert out.dtype == torch.bfloat16 and {p.dtype for p in network.trainable_parameters()} == {torch.float32}


def test_the_state_dict_holds_comfyui_keys_with_the_alpha() -> None:
    network = VisageLoRANetwork(tiny_transformer(), rank=2, alpha=4.0)

    state = network.get_state_dict(torch.float16)

    assert state["diffusion_model.blocks.0.attn.qkv.lora_A.weight"].shape == (2, 128)
    assert state["diffusion_model.blocks.0.attn.qkv.lora_B.weight"].shape == (3 * 128, 2)
    assert state["diffusion_model.blocks.0.attn.qkv.lora_B.weight"].dtype == torch.float16
    assert state["diffusion_model.blocks.0.attn.qkv.alpha"].item() == 4.0
    assert len(state) == 3 * len(network.lora_layers)


def test_the_prefix_and_the_unconverted_dtype_are_honoured() -> None:
    network = VisageLoRANetwork(tiny_transformer(), rank=2, prefix="transformer.")

    state = network.get_state_dict(None)

    assert state["transformer.blocks.0.attn.qkv.lora_A.weight"].dtype == torch.float32


def test_the_optimizer_groups_split_into_attention_mlp_and_other() -> None:
    network = VisageLoRANetwork(tiny_transformer(), rank=2, targets="all")

    groups = {g["name"]: g["params"] for g in network.param_groups(split=True)}
    whole = network.param_groups(split=False)

    assert set(groups) == {"attn", "mlp", "other"}
    assert sum(len(p) for p in groups.values()) == len(whole[0]["params"]) == 2 * len(network.lora_layers)
    assert whole[0]["name"] == "lora"


def test_the_lora_dtype_is_applied_to_the_adapters_only() -> None:
    model = tiny_transformer()
    network = VisageLoRANetwork(model, rank=2)

    network.set_lora_dtype("cpu", torch.bfloat16)

    assert {p.dtype for p in network.trainable_parameters()} == {torch.bfloat16}
    assert {p.dtype for p in model.parameters()} == {torch.float32}


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("blocks.0.attn.qkv", "attn"),
        ("blocks.3.attn.out_proj", "attn"),
        ("blocks.0.mlp.fc1", "mlp"),
        ("blocks.17.mlp.fc2", "mlp"),
        ("blocks.0.mod.1", "other"),
        ("text_proj", "other"),
    ],
)
def test_modules_group_by_family(name: str, group: str) -> None:
    assert param_group_of(name) == group


def test_an_unknown_preset_is_refused() -> None:
    with pytest.raises(ValueError, match=r"lora\.targets must be one of"):
        VisageLoRANetwork(tiny_transformer(), targets="everything")
    with pytest.raises(ValueError, match=r"lora\.targets must be one of"):
        selected("blocks.0.attn.qkv", "everything")


def test_a_model_without_matching_linears_is_refused() -> None:
    with pytest.raises(ValueError, match=r"No Visage-Image modules matched"):
        VisageLoRANetwork(torch.nn.Sequential(torch.nn.Linear(2, 2)), rank=2)
