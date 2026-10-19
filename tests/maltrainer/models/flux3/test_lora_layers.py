from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.flux3.lora import Flux3LoRANetwork, LoRALayer, param_group_of, selected
from maltrainer.models.flux3.model import Flux3Config

FUSED = Flux3Config(hidden_size=128, num_heads=1, depth=1, depth_single_blocks=2, fused_qkv=True)


@pytest.mark.parametrize(
    ("name", "preset", "expected"),
    [
        ("single_blocks.3.q_proj", "blocks", True),
        ("txt_mode_blocks.0.mlp_in", "blocks", True),
        ("content_mode_blocks.image.2.attn_out", "blocks", True),
        ("content_mode_blocks.image_cond.0.q_proj", "blocks", False),
        ("single_blocks.3.q_proj", "single", True),
        ("txt_mode_blocks.0.q_proj", "single", False),
        ("single_blocks.3.norm.query_norm", "blocks", False),
        ("txt_in", "blocks", False),
        ("txt_in", "all", True),
        ("final_layer.image.adaLN_modulation.1", "all", True),
        ("early_stream_modulations.image_cond.lin", "all", False),
        ("ref_time_in.in_layer", "all", False),
    ],
)
def test_a_linear_is_adapted_when_its_name_is_in_the_preset(name: str, preset: str, expected: bool) -> None:
    assert selected(name, preset) is expected


def test_an_unknown_preset_is_refused_with_the_list_of_valid_ones() -> None:
    with pytest.raises(ValueError, match="lora.targets must be one of .*'blocks', 'single', 'all'"):
        selected("single_blocks.0.q_proj", "everything")


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("single_blocks.0.k_proj", "attn"),
        ("txt_mode_blocks.0.attn_out", "attn"),
        ("a.mlp_in", "mlp"),
        ("txt_in", "other"),
    ],
)
def test_each_module_family_gets_its_own_learning_rate_group(name: str, group: str) -> None:
    assert param_group_of(name) == group


def test_a_layer_is_an_exact_no_op_at_init_and_then_adds_up_of_down_scaled_by_alpha_over_rank() -> None:
    torch.manual_seed(0)
    layer = LoRALayer(3, 5, rank=2, alpha=6.0)
    x = torch.randn(4, 3)
    assert torch.equal(layer(x), torch.zeros(4, 5))

    layer.lora_up.weight.data.normal_()

    expected = x @ layer.lora_down.weight.T @ layer.lora_up.weight.T * 3.0
    torch.testing.assert_close(layer(x), expected)
    assert (layer.lora_down.weight.shape, layer.lora_up.weight.shape) == ((2, 3), (5, 2))


def test_a_layer_casts_the_input_to_its_own_dtype_outside_autocast() -> None:
    layer = LoRALayer(3, 2, rank=1, alpha=1.0).to(torch.bfloat16)

    assert layer(torch.ones(1, 3)).dtype == torch.bfloat16


def test_a_layer_leaves_the_input_dtype_to_autocast_when_it_is_active(monkeypatch: pytest.MonkeyPatch) -> None:
    layer = LoRALayer(3, 2, rank=1, alpha=1.0)
    seen = []

    def widen(_module: nn.Module, args: tuple) -> tuple:
        seen.append(args[0].dtype)
        return (args[0].float(),)

    layer.lora_down.register_forward_pre_hook(widen)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_: True)

    layer(torch.ones(1, 3, dtype=torch.bfloat16))

    assert seen == [torch.bfloat16]


def test_a_wrapped_linear_returns_its_output_plus_the_adapter_delta() -> None:
    torch.manual_seed(0)
    linear, layer = nn.Linear(3, 4, bias=False), LoRALayer(3, 4, rank=2, alpha=2.0)
    layer.lora_up.weight.data.normal_()
    x = torch.randn(5, 3)
    base = linear(x)

    Flux3LoRANetwork._wrap(linear, [layer])

    torch.testing.assert_close(linear(x), base + layer(x))


def test_a_fused_projection_gets_one_delta_slice_per_adapter() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(3, 6, bias=False)
    layers = [LoRALayer(3, 2, rank=1, alpha=1.0) for _ in range(3)]
    for layer in layers:
        layer.lora_up.weight.data.normal_()
    x = torch.randn(4, 3)
    base = linear(x)

    Flux3LoRANetwork._wrap(linear, layers, concat=True)

    torch.testing.assert_close(linear(x), base + torch.cat([layer(x) for layer in layers], dim=-1))


def test_a_float32_adapter_on_a_bf16_base_returns_the_base_dtype() -> None:
    linear = nn.Linear(3, 2, bias=False).to(torch.bfloat16)
    Flux3LoRANetwork._wrap(linear, [LoRALayer(3, 2, rank=1, alpha=1.0)])

    assert linear(torch.ones(1, 3, dtype=torch.bfloat16)).dtype == torch.bfloat16
