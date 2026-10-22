from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRANetwork
from tests.maltrainer.models.minimax_h3.h3_tiny import tiny_transformer

PER_BLOCK = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
LOGGER = "maltrainer.models.minimax_h3.lora"


def _network(**options: object) -> MiniMaxH3LoRANetwork:
    torch.manual_seed(0)
    return MiniMaxH3LoRANetwork(tiny_transformer(), rank=2, alpha=4.0, **options)


def _paths(blocks: list[int], refiner: bool = True, suffixes: tuple[str, ...] = PER_BLOCK) -> list[str]:
    names = [f"blocks.{index}.{suffix}" for index in blocks for suffix in suffixes]
    if refiner:
        names += [f"token_refiner.blocks.0.{suffix}" for suffix in suffixes]
    return sorted(names)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.plain = nn.ModuleDict({"qkv_proj": nn.Linear(2, 2)})
        self.norm_stage = nn.ModuleDict({"qkv_proj": nn.Linear(2, 2)})
        self.modulation_stage = nn.ModuleDict({"qkv_proj": nn.Linear(2, 2)})
        self.qkv_proj = nn.Identity()


def test_the_default_targets_are_the_attention_and_mlp_linears_of_every_block_and_the_refiner() -> None:
    network = _network()

    assert sorted(network.name_map.values()) == _paths([0, 1, 2])
    assert len(network.lora_layers) == 16
    assert network.name_map["blocks_0_attn_qkv_proj"] == "blocks.0.attn.qkv_proj"
    assert set(network.lora_layers) == set(network.name_map)


def test_the_layers_take_the_networks_rank_and_alpha_and_the_prefix_defaults_to_diffusion_model() -> None:
    network = _network()

    assert network.prefix == "diffusion_model."
    assert all(layer.rank == 2 and layer.alpha == 4.0 and layer.scale == 2.0 for layer in network.lora_layers.values())


def test_a_block_range_is_inclusive_and_leaves_the_refiner_alone() -> None:
    network = _network(block_range=(1, 1))

    assert sorted(network.name_map.values()) == _paths([1])


def test_a_block_range_given_as_a_list_is_stored_as_a_tuple() -> None:
    network = _network(block_range=[1, 2], include_token_refiner=False)

    assert network.block_range == (1, 2)
    assert sorted(network.name_map.values()) == _paths([1, 2], refiner=False)


def test_the_refiner_can_be_left_out() -> None:
    network = _network(include_token_refiner=False)

    assert sorted(network.name_map.values()) == _paths([0, 1, 2], refiner=False)


def test_custom_targets_match_the_last_component_of_a_module_path() -> None:
    network = _network(target_modules=["fc1"])

    assert sorted(network.name_map.values()) == _paths([0, 1, 2], suffixes=("mlp.fc1",))


def test_a_target_that_is_only_part_of_a_name_catches_nothing() -> None:
    with pytest.raises(ValueError, match="No MiniMax H3 modules matched"):
        _network(target_modules=["proj"])


def test_a_block_range_with_no_block_in_it_and_no_refiner_is_rejected() -> None:
    with pytest.raises(ValueError, match="block_range"):
        _network(block_range=(10, 12), include_token_refiner=False)


def test_adaln_projections_of_the_blocks_are_adapted_only_on_request() -> None:
    plain, with_adaln = _network(), _network(include_adaln=True)

    assert not any("adaln" in name for name in plain.name_map.values())
    assert sorted(set(with_adaln.name_map.values()) - set(plain.name_map.values())) == [
        f"blocks.{index}.adaln_proj.linear" for index in range(3)
    ]
    layer = with_adaln.lora_layers["blocks_1_adaln_proj_linear"]
    assert (layer.lora_down.weight.shape, layer.lora_up.weight.shape) == ((2, 10), (288, 2))


def test_the_final_layer_adaln_is_never_adapted_and_adaln_follows_the_block_range() -> None:
    network = _network(include_adaln=True, block_range=(2, 2), include_token_refiner=False)

    assert sorted(network.name_map.values()) == _paths([2], refiner=False, suffixes=(*PER_BLOCK, "adaln_proj.linear"))


def test_paths_containing_norm_or_modulation_are_never_adapted() -> None:
    network = MiniMaxH3LoRANetwork(_Toy(), rank=1, alpha=1.0)

    assert list(network.name_map.values()) == ["plain.qkv_proj"]


def test_injection_is_logged_with_its_count_and_block_range(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger=LOGGER):
        _network(block_range=(1, 2), include_adaln=True, include_token_refiner=False)

    assert "Injected 10 LoRA layers" in caplog.text
    assert "including 2 per-block adaln_proj" in caplog.text
    assert "blocks 1-2" in caplog.text
    assert "token refiner frozen" in caplog.text
