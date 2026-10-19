from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.supra2_img.lora import SupraLoRANetwork
from maltrainer.models.supra2_img.model import SupraDiT

DEPTH, RANK = 2, 4


def _network(alpha: float = 8.0, prefix: str = "", targets: list[str] | None = None) -> SupraLoRANetwork:
    torch.manual_seed(0)
    dit = SupraDiT(latent_ch=4, d_model=32, depth=DEPTH, n_heads=4, ctx_dim=16, num_tokens=16)
    network = SupraLoRANetwork(dit, rank=RANK, alpha=alpha, target_modules=targets, prefix=prefix)
    for layer in network.lora_layers.values():
        layer.lora_up.weight.data.normal_()
    return network


def _save(tmp_path: Path, state: dict[str, torch.Tensor]) -> Path:
    path = tmp_path / "lora.safetensors"
    save_file(state, str(path))
    return path


def test_the_state_dict_holds_a_down_up_and_alpha_entry_per_layer_under_the_prefix() -> None:
    network = _network(prefix="transformer.", targets=["cross_attn.kv"])

    state = network.get_state_dict()

    assert sorted(state) == [
        f"transformer.blocks.{block}.cross_attn.kv.{suffix}"
        for block in range(DEPTH)
        for suffix in ("alpha", "lora_A.weight", "lora_B.weight")
    ]


def test_saved_weights_are_cast_to_bfloat16_by_default_and_alpha_stays_float32() -> None:
    network = _network(alpha=6.0, targets=["mlp.2"])

    state = network.get_state_dict()

    down, up, alpha = (state[f"blocks.0.mlp.2.{k}"] for k in ("lora_A.weight", "lora_B.weight", "alpha"))
    assert (down.dtype, up.dtype, alpha.dtype) == (torch.bfloat16, torch.bfloat16, torch.float32)
    assert down.shape == (RANK, 128) and up.shape == (32, RANK)
    assert alpha.item() == 6.0


def test_the_save_dtype_can_be_kept_as_trained_and_the_tensors_are_detached() -> None:
    network = _network(targets=["mlp.0"])

    state = network.get_state_dict(save_dtype=None)

    saved = state["blocks.1.mlp.0.lora_B.weight"]
    assert saved.dtype == torch.float32 and not saved.requires_grad
    assert torch.equal(saved, network.lora_layers["blocks_1_mlp_0"].lora_up.weight)


def test_saved_weights_load_back_into_a_fresh_network_with_the_stored_alpha(tmp_path: Path) -> None:
    source = _network(alpha=6.0, prefix="p.")
    path = _save(tmp_path, source.get_state_dict(save_dtype=None))
    target = _network(alpha=1.0, prefix="p.")
    torch.manual_seed(5)

    loaded = target.load_weights(path)

    assert loaded == len(source.lora_layers)
    for name, layer in target.lora_layers.items():
        assert torch.equal(layer.lora_down.weight, source.lora_layers[name].lora_down.weight)
        assert torch.equal(layer.lora_up.weight, source.lora_layers[name].lora_up.weight)
        assert (layer.alpha, layer.scale) == (6.0, 6.0 / RANK)


def test_weights_saved_in_bfloat16_are_cast_to_the_adapters_dtype(tmp_path: Path) -> None:
    source = _network(targets=["mlp.0"])
    target = _network(targets=["mlp.0"])

    target.load_weights(_save(tmp_path, source.get_state_dict()))

    layer = target.lora_layers["blocks_0_mlp_0"]
    assert layer.lora_up.weight.dtype == torch.float32
    torch.testing.assert_close(
        layer.lora_up.weight, source.lora_layers["blocks_0_mlp_0"].lora_up.weight.bfloat16().float()
    )


def test_a_file_without_alpha_keeps_the_networks_alpha(tmp_path: Path) -> None:
    source = _network(alpha=6.0, targets=["mlp.0"])
    state = {key: value for key, value in source.get_state_dict().items() if not key.endswith(".alpha")}
    target = _network(alpha=2.0, targets=["mlp.0"])

    target.load_weights(_save(tmp_path, state))

    assert {layer.alpha for layer in target.lora_layers.values()} == {2.0}


@pytest.mark.parametrize("missing", ["lora_A.weight", "lora_B.weight"])
def test_a_file_missing_part_of_a_layer_is_an_error_naming_the_layer(tmp_path: Path, missing: str) -> None:
    state = _network(targets=["mlp.0"]).get_state_dict()
    del state[f"blocks.1.mlp.0.{missing}"]

    with pytest.raises(RuntimeError, match=r"missing 1 layers, including blocks.1.mlp.0"):
        _network(targets=["mlp.0"]).load_weights(_save(tmp_path, state))


def test_a_file_saved_under_another_prefix_matches_no_layer(tmp_path: Path) -> None:
    state = _network(prefix="a.", targets=["mlp.0"]).get_state_dict()

    with pytest.raises(RuntimeError, match=r"missing 2 layers, including blocks\.0\.mlp\.0, blocks\.1\.mlp\.0"):
        _network(prefix="", targets=["mlp.0"]).load_weights(_save(tmp_path, state))
