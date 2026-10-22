"""The Krea 2 LoRA network: which Linears it targets, how it wraps them and what it saves."""

from __future__ import annotations

import re

import pytest
import torch

from maltrainer.models.krea2.lora import Krea2LoRANetwork, LoRALayer, _selected, param_group_of

from .tiny import inputs, tiny_model

# tiny model: 2 blocks x (5 attention + 3 MLP), 4 text-fusion blocks x 8, 1 projector, 7 I/O layers
COUNTS = {"attn-mlp": 16, "blocks+txt": 16 + 32, "all": 16 + 32 + 1 + 7}


@pytest.mark.parametrize("preset", COUNTS)
def test_each_target_preset_wraps_its_documented_set_of_linears(preset):
    network = Krea2LoRANetwork(tiny_model(), targets=preset)
    assert len(network.lora_layers) == COUNTS[preset]


def test_unknown_target_preset_is_refused():
    with pytest.raises(ValueError, match=re.escape("lora.targets must be one of")):
        Krea2LoRANetwork(tiny_model(), targets="everything")


def test_a_block_range_matching_nothing_is_an_error():
    with pytest.raises(ValueError, match="No Krea 2 modules matched"):
        Krea2LoRANetwork(tiny_model(), block_range=(5, 9))


def test_block_range_keeps_only_those_blocks_but_not_the_text_fusion():
    network = Krea2LoRANetwork(tiny_model(), targets="blocks+txt", block_range=[1, 1])
    names = set(network.name_map.values())
    assert {n for n in names if n.startswith("blocks.")} == {
        f"blocks.1.{leaf}"
        for leaf in ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down")
    }
    assert sum(n.startswith("txtfusion.") for n in names) == 32


@pytest.mark.parametrize(
    "name, preset, expected",
    [
        ("blocks.3.attn.wq", "attn-mlp", True),
        ("blocks.3.tproj", "attn-mlp", False),
        ("txtfusion.projector", "blocks+txt", False),
        ("txtfusion.projector", "all", True),
        ("txtfusion.layerwise_blocks.0.attn.wq", "attn-mlp", False),
        ("txtfusion.refiner_blocks.1.mlp.down", "blocks+txt", True),
        ("last.linear", "blocks+txt", False),
        ("last.linear", "all", True),
        ("txtmlp.1", "all", True),
        ("txtmlp.2", "all", False),
    ],
)
def test_selection_rules(name, preset, expected):
    assert _selected(name, preset, None) is expected


@pytest.mark.parametrize(
    "name, group",
    [
        ("txtfusion.refiner_blocks.0.attn.wq", "txtfusion"),
        ("blocks.0.attn.wo", "attn"),
        ("blocks.0.mlp.up", "mlp"),
        ("last.linear", "other"),
    ],
)
def test_param_group_names(name, group):
    assert param_group_of(name) == group


def test_lora_layer_scale_is_alpha_over_rank_and_starts_as_a_no_op():
    layer = LoRALayer(8, 6, rank=4, alpha=2.0)
    assert layer.scale == 0.5
    assert layer.lora_up.weight.eq(0).all() and layer.lora_down.weight.any()
    assert layer(torch.randn(3, 8)).eq(0).all()


def test_lora_layer_output_is_up_down_x_times_scale():
    layer = LoRALayer(2, 2, rank=1, alpha=3.0)  # scale 3
    layer.lora_down.weight.data = torch.tensor([[1.0, 2.0]])
    layer.lora_up.weight.data = torch.tensor([[1.0], [-1.0]])
    assert layer(torch.tensor([[1.0, 1.0]])).tolist() == [[9.0, -9.0]]


def test_lora_layer_casts_input_to_its_weight_dtype():
    layer = LoRALayer(4, 4, rank=2, alpha=2.0).to(torch.bfloat16)
    assert layer(torch.randn(2, 4)).dtype == torch.bfloat16


def test_fresh_network_leaves_the_model_output_unchanged():
    model = tiny_model()
    x, sigma, context = inputs(batch=1)
    base = model(x, sigma, context)
    Krea2LoRANetwork(model, targets="all")
    assert torch.equal(model(x, sigma, context), base)


def test_trained_adapter_adds_its_delta_to_the_wrapped_linear_and_disabled_skips_it():
    model = tiny_model()
    network = Krea2LoRANetwork(model, rank=2, alpha=4.0)
    layer = network.lora_layers["blocks_0_attn_wq"]
    layer.lora_up.weight.data.normal_()
    linear = model.blocks[0].attn.wq
    x = torch.randn(3, 64)
    delta = layer(x)
    with network.disabled():
        base = linear(x)
    assert delta.abs().sum() > 0
    assert torch.allclose(linear(x), base + delta, atol=1e-6)
    assert not torch.allclose(linear(x), base)  # re-enabled after the block exits


def test_wrapped_output_keeps_the_base_dtype():
    model = tiny_model()
    network = Krea2LoRANetwork(model)
    network.set_lora_dtype("cpu", torch.bfloat16)
    out = model.blocks[0].mlp.gate(torch.randn(2, 64))
    assert out.dtype == torch.float32  # the bf16 delta is cast to the base Linear's dtype


def test_forward_delegates_to_the_transformer():
    model = tiny_model()
    network = Krea2LoRANetwork(model)
    x, sigma, context = inputs(batch=1)
    assert torch.equal(network(x, sigma, context), model(x, sigma, context))


def test_state_dict_uses_comfy_keys_with_alpha_and_requested_dtype():
    network = Krea2LoRANetwork(tiny_model(), rank=2, alpha=8.0)
    state = network.get_state_dict()
    key = "diffusion_model.blocks.1.mlp.down"
    assert state[f"{key}.lora_A.weight"].shape == (2, 128)  # (rank, in): mlp hidden is 128
    assert state[f"{key}.lora_B.weight"].shape == (64, 2)
    assert state[f"{key}.lora_A.weight"].dtype == torch.bfloat16
    assert state[f"{key}.alpha"].item() == 8.0 and state[f"{key}.alpha"].dtype == torch.float32
    assert torch.equal(
        state[f"{key}.lora_A.weight"], network.lora_layers["blocks_1_mlp_down"].lora_down.weight.detach().bfloat16()
    )
    assert len(state) == 3 * len(network.lora_layers)


def test_state_dict_without_a_save_dtype_keeps_training_precision_and_honours_prefix():
    network = Krea2LoRANetwork(tiny_model(), prefix="lora_unet.")
    state = network.get_state_dict(save_dtype=None)
    assert all(k.startswith("lora_unet.blocks.") for k in state)
    assert state["lora_unet.blocks.0.attn.wq.lora_A.weight"].dtype == torch.float32
