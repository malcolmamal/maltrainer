from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRANetwork
from tests.maltrainer.models.minimax_h3.h3_adapters import KOHYA, hand_pair, write_adapter
from tests.maltrainer.models.minimax_h3.h3_tiny import tiny_transformer

MODULE = "blocks.0.mlp.fc1"
LOGGER = "maltrainer.models.minimax_h3.lora"


def _network() -> MiniMaxH3LoRANetwork:
    return MiniMaxH3LoRANetwork(tiny_transformer(), rank=2, alpha=4.0, block_range=(1, 1))


def _fc1(network: MiniMaxH3LoRANetwork) -> torch.nn.Linear:
    return network.transformer.blocks[0].mlp.fc1


def _delta(x: torch.Tensor, down: torch.Tensor, up: torch.Tensor, scale: float) -> torch.Tensor:
    return x @ down.T @ up.T * scale


def test_an_adapter_adds_its_delta_to_the_module_scaled_by_alpha_over_rank_times_the_multiplier(tmp_path: Path) -> None:
    network = _network()
    down, up = hand_pair(16, 16)
    path = write_adapter(tmp_path / "assist.safetensors", {MODULE: (down, up, 4.0)})
    x = torch.randn(3, 16)

    attached = network.load_assistant_adapter(str(path), multiplier=0.5)

    assert attached == 1
    expected = F.linear(x, _fc1(network).weight) + _delta(x, down, up, (4.0 / 2) * 0.5)
    torch.testing.assert_close(_fc1(network)(x), expected)


def test_an_adapter_without_an_alpha_has_a_scale_of_one_times_the_multiplier(tmp_path: Path) -> None:
    network = _network()
    down, up = hand_pair(16, 16)
    path = write_adapter(tmp_path / "assist.safetensors", {MODULE: (down, up, None)})
    x = torch.randn(2, 16)

    network.load_assistant_adapter(str(path), multiplier=2.0)

    torch.testing.assert_close(_fc1(network)(x), F.linear(x, _fc1(network).weight) + _delta(x, down, up, 2.0))


@pytest.mark.parametrize("style", ["peft", "kohya", "unprefixed"])
def test_peft_kohya_and_unprefixed_names_address_the_same_module(tmp_path: Path, style: str) -> None:
    network = _network()
    down, up = hand_pair(16, 16)
    options = {"peft": {}, "kohya": {"names": KOHYA, "kohya_names": True}, "unprefixed": {"prefix": ""}}[style]
    path = write_adapter(tmp_path / "assist.safetensors", {MODULE: (down, up, 2.0)}, **options)
    x = torch.randn(2, 16)

    network.load_assistant_adapter(str(path))

    torch.testing.assert_close(_fc1(network)(x), F.linear(x, _fc1(network).weight) + _delta(x, down, up, 1.0))


def test_the_assistant_is_frozen_and_kept_out_of_the_trainable_set_and_the_saved_state(tmp_path: Path) -> None:
    network = _network()
    before = (len(list(network.trainable_parameters())), len(network.get_state_dict()))
    path = write_adapter(tmp_path / "assist.safetensors", {MODULE: (*hand_pair(16, 16), 2.0)})

    network.load_assistant_adapter(str(path))

    layer = network.assistant_layers["blocks_0_mlp_fc1"]
    assert not any(p.requires_grad for p in layer.parameters())
    assert not layer.training
    assert (len(list(network.trainable_parameters())), len(network.get_state_dict())) == before
    assert "blocks_0_mlp_fc1" not in network.lora_layers


def test_the_trainable_adapter_and_the_assistant_both_act_on_the_same_linear(tmp_path: Path) -> None:
    network = MiniMaxH3LoRANetwork(tiny_transformer(), rank=2, alpha=4.0, block_range=(0, 0))
    trained = network.lora_layers["blocks_0_mlp_fc1"]
    torch.nn.init.normal_(trained.lora_up.weight)
    down, up = hand_pair(16, 16)
    path = write_adapter(tmp_path / "assist.safetensors", {MODULE: (down, up, 2.0)})
    x = torch.randn(2, 16)

    network.load_assistant_adapter(str(path))

    expected = F.linear(x, _fc1(network).weight) + trained(x) + _delta(x, down, up, 1.0)
    torch.testing.assert_close(_fc1(network)(x), expected)


def test_a_down_matrix_without_its_up_partner_is_ignored(tmp_path: Path) -> None:
    network = _network()
    path = write_adapter(tmp_path / "assist.safetensors", {MODULE: (*hand_pair(16, 16), 2.0)})
    tensors = {key: value for key, value in load_file(str(path)).items() if "lora_B" not in key}
    save_file(tensors, str(path))
    x = torch.randn(2, 16)

    attached = network.load_assistant_adapter(str(path))

    assert attached == 0
    torch.testing.assert_close(_fc1(network)(x), F.linear(x, _fc1(network).weight))


def test_the_adapter_layers_can_be_cast_to_another_dtype(tmp_path: Path) -> None:
    network = _network()
    path = write_adapter(tmp_path / "assist.safetensors", {MODULE: (*hand_pair(16, 16), 2.0)})

    network.load_assistant_adapter(str(path), device=torch.device("cpu"), dtype=torch.bfloat16)

    assert network.assistant_layers["blocks_0_mlp_fc1"].lora_down.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("target", ["blocks.0.nonexistent", "blocks.0.mlp", "blocks.9.mlp.fc1"])
def test_an_adapter_aimed_at_a_module_this_transformer_lacks_is_rejected(tmp_path: Path, target: str) -> None:
    network = _network()
    path = write_adapter(tmp_path / "assist.safetensors", {target: (*hand_pair(16, 16), 2.0)})

    with pytest.raises(ValueError, match=f"does not expose.*{target}"):
        network.load_assistant_adapter(str(path))


def test_a_down_matrix_of_the_wrong_width_for_an_ordinary_module_is_rejected(tmp_path: Path) -> None:
    network = _network()
    path = write_adapter(tmp_path / "assist.safetensors", {MODULE: (*hand_pair(5, 16), 2.0)})

    with pytest.raises(ValueError, match=r"in 5 != 16"):
        network.load_assistant_adapter(str(path))


def test_the_attachment_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    network = _network()
    path = write_adapter(tmp_path / "assist.safetensors", {MODULE: (*hand_pair(16, 16), 2.0)})

    with caplog.at_level("INFO", logger=LOGGER):
        network.load_assistant_adapter(str(path), multiplier=0.25)

    assert "Attached frozen training adapter: 1 modules (0 AdaLN via the t_emb grid), rank 2, multiplier 0.25" in caplog.text
    assert "assist.safetensors" in caplog.text


def test_tensors_that_are_not_lora_weights_are_ignored(tmp_path: Path) -> None:
    network = _network()
    path = write_adapter(tmp_path / "assist.safetensors", {MODULE: (*hand_pair(16, 16), 2.0)})
    save_file({**load_file(str(path)), "diffusion_model.blocks.0.mlp.fc1.weight": torch.zeros(1)}, str(path))

    assert network.load_assistant_adapter(str(path)) == 1


def test_a_second_adapter_attaches_next_to_the_first(tmp_path: Path) -> None:
    network = _network()
    first = write_adapter(tmp_path / "first.safetensors", {MODULE: (*hand_pair(16, 16), 2.0)})
    second = write_adapter(tmp_path / "second.safetensors", {"blocks.0.mlp.fc2": (*hand_pair(8, 16), 2.0)})

    network.load_assistant_adapter(str(first))
    network.load_assistant_adapter(str(second))

    assert sorted(network.assistant_layers) == ["blocks_0_mlp_fc1", "blocks_0_mlp_fc2"]
