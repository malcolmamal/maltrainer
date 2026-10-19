from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.krea2.lora import Krea2LoRANetwork, LoRALayer, _selected, param_group_of
from tests.maltrainer.models.krea2.tiny_model import tiny_transformer


@pytest.mark.parametrize(
    ("name", "preset", "block_range", "expected"),
    [
        ("blocks.3.attn.wq", "attn-mlp", None, True),
        ("blocks.3.attn.gate", "attn-mlp", None, True),
        ("blocks.3.mlp.down", "attn-mlp", None, True),
        ("blocks.3.attn.qknorm", "attn-mlp", None, False),
        ("blocks.3.attn.wq", "attn-mlp", (0, 2), False),
        ("blocks.2.attn.wq", "attn-mlp", (0, 2), True),
        ("blocks.0.attn.wq", "attn-mlp", (1, 2), False),
        ("txtfusion.layerwise_blocks.0.attn.wq", "attn-mlp", None, False),
        ("txtfusion.layerwise_blocks.0.attn.wq", "blocks+txt", None, True),
        ("txtfusion.projector", "blocks+txt", None, False),
        ("txtfusion.projector", "all", None, True),
        ("txtfusion.refiner_blocks.1.mlp.up", "all", (5, 6), True),
        ("first", "blocks+txt", None, False),
        ("first", "all", None, True),
        ("last.linear", "all", None, True),
        ("txtmlp.2", "all", None, False),
    ],
)
def test_a_linear_is_adapted_when_its_name_preset_and_block_index_allow_it(
    name: str, preset: str, block_range: tuple | None, expected: bool
) -> None:
    assert _selected(name, preset, block_range) is expected


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("txtfusion.refiner_blocks.0.attn.wq", "txtfusion"),
        ("blocks.0.attn.wo", "attn"),
        ("blocks.0.mlp.up", "mlp"),
        ("tproj.1", "other"),
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


@pytest.mark.parametrize(
    ("targets", "block_range", "count"),
    [
        ("attn-mlp", None, 16),
        ("blocks+txt", None, 16 + 32),
        ("all", None, 16 + 32 + 1 + 7),
        ("attn-mlp", [1, 1], 8),
        ("blocks+txt", (0, 0), 8 + 32),
    ],
)
def test_each_preset_adapts_the_expected_number_of_linears(targets: str, block_range: object, count: int) -> None:
    network = Krea2LoRANetwork(tiny_transformer(), rank=2, targets=targets, block_range=block_range)

    assert len(network.lora_layers) == len(network.name_map) == count


def test_the_block_range_is_inclusive_on_both_ends() -> None:
    network = Krea2LoRANetwork(tiny_transformer(), rank=2, block_range=[1, 1])

    assert {name.split(".")[1] for name in network.name_map.values()} == {"1"}
    assert network.name_map["blocks_1_attn_gate"] == "blocks.1.attn.gate"


def test_a_target_set_that_matches_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="No Krea 2 modules matched"):
        Krea2LoRANetwork(tiny_transformer(), block_range=(10, 12))


def test_an_unknown_preset_is_refused() -> None:
    with pytest.raises(ValueError, match="lora.targets must be one of"):
        Krea2LoRANetwork(tiny_transformer(), targets="blocks")
