from __future__ import annotations

import re
from typing import Callable, List

import pytest
import torch

from maltrainer.models.moss_image.lora import MOSSImageLoRANetwork
from maltrainer.models.moss_image.model import MOSSImageTransformer

Maker = Callable[..., MOSSImageTransformer]
BLOCK_LEAVES = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
IO = [
    "video_embedder",
    "time_embedding.0",
    "time_embedding.2",
    "shared_block_modulation",
    "video_head_modulation",
    "final_linear_video",
]


def _layers(*blocks: int, leaves=BLOCK_LEAVES) -> List[str]:
    return sorted(f"layers.{i}.proj.{leaf}" for i in blocks for leaf in leaves)


def _train(network: MOSSImageLoRANetwork, seed: int = 0) -> None:
    torch.manual_seed(seed)
    for layer in network.lora_layers.values():
        layer.lora_up.weight.data.normal_(std=0.5)


def _run(model: MOSSImageTransformer) -> torch.Tensor:
    torch.manual_seed(4)
    return model(torch.randn(1, 128, 2, 2), torch.full((1,), 0.4), torch.randn(1, 3, 8))


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        ("blocks", _layers(0, 1)),
        ("attn", _layers(0, 1, leaves=("q", "k", "v", "o"))),
        ("all", sorted(_layers(0, 1) + IO)),
    ],
)
def test_each_target_preset_adapts_its_linears_and_names_a_fused_qkv_as_q_k_v(
    make_model: Maker, fused: bool, targets: str, expected: List[str]
) -> None:
    network = MOSSImageLoRANetwork(make_model(fused_qkv=fused), rank=2, targets=targets)

    assert sorted(network.name_map.values()) == expected
    assert sorted(network.lora_layers) == sorted(name.replace(".", "_") for name in expected)


def test_a_fresh_network_leaves_the_models_output_exactly_unchanged(make_model: Maker) -> None:
    model = make_model()
    before = _run(model)

    network = MOSSImageLoRANetwork(model, rank=4, alpha=8.0, targets="all")

    assert torch.equal(_run(model), before)
    assert torch.equal(_run(network), before)


def _delta(network: MOSSImageLoRANetwork, name: str, x: torch.Tensor, scale: float) -> torch.Tensor:
    layer = network.lora_layers[name]
    return x @ layer.lora_down.weight.T @ layer.lora_up.weight.T * scale


@pytest.mark.parametrize("fused", [False, True])
def test_an_adapted_linear_adds_up_of_down_times_alpha_over_rank_to_its_base_output(
    make_model: Maker, fused: bool
) -> None:
    model = make_model(fused_qkv=fused)
    linear = model.layers[0].proj["qkv" if fused else "q"]
    x = torch.randn(3, 128)
    base = linear(x)
    network = MOSSImageLoRANetwork(model, rank=2, alpha=6.0, targets="attn")
    _train(network)

    out = linear(x)

    names = ("q", "k", "v") if fused else ("q",)
    delta = torch.cat([_delta(network, f"layers_0_proj_{n}", x, 3.0) for n in names], dim=-1)
    torch.testing.assert_close(out, base + delta)
    assert delta.abs().max() > 0


def test_a_fused_qkv_adapter_is_split_into_three_thirds_of_the_output(make_model: Maker) -> None:
    network = MOSSImageLoRANetwork(make_model(fused_qkv=True), rank=2, targets="attn")

    shapes = {n: network.lora_layers[f"layers_1_proj_{n}"].lora_up.weight.shape for n in "qkvo"}

    assert shapes == dict.fromkeys("qkvo", (128, 2))
    assert network.lora_layers["layers_1_proj_q"].lora_down.weight.shape == (2, 128)


def test_skipped_blocks_get_no_adapter_and_the_list_is_sorted_and_deduplicated(make_model: Maker) -> None:
    network = MOSSImageLoRANetwork(make_model(num_layers=3), rank=2, skip_blocks=[2, 0, 2])

    assert network.skip_blocks == (0, 2)
    assert sorted(network.name_map.values()) == _layers(1)


def test_skipping_block_1_does_not_skip_block_10(make_model: Maker) -> None:
    network = MOSSImageLoRANetwork(make_model(num_layers=11), rank=2, targets="attn", skip_blocks=[1])

    assert "layers.10.proj.q" in network.name_map.values()
    assert "layers.1.proj.q" not in network.name_map.values()


def test_skipping_every_block_of_the_blocks_preset_leaves_nothing_to_train(make_model: Maker) -> None:
    with pytest.raises(ValueError, match="No MOSS-Image modules matched"):
        MOSSImageLoRANetwork(make_model(), skip_blocks=[0, 1])


def test_an_unknown_target_preset_is_rejected_before_anything_is_wrapped(make_model: Maker) -> None:
    model = make_model()
    before = _run(model)

    with pytest.raises(ValueError, match=re.escape("lora.targets must be one of")):
        MOSSImageLoRANetwork(model, targets="everything")

    assert not any("forward" in vars(m) for m in model.modules())
    assert torch.equal(_run(model), before)


def test_a_bf16_base_with_fp32_adapters_keeps_the_base_dtype(make_model: Maker) -> None:
    model = make_model().to(torch.bfloat16)
    model.compute_dtype = torch.bfloat16
    network = MOSSImageLoRANetwork(model, rank=2)
    network.set_lora_dtype(torch.device("cpu"), torch.float32)
    _train(network)

    out = model.layers[0].proj["q"](torch.randn(2, 128, dtype=torch.bfloat16))

    assert out.dtype == torch.bfloat16
    assert {p.dtype for p in network.trainable_parameters()} == {torch.float32}
    assert _run(model).dtype == torch.bfloat16


def test_only_the_adapters_receive_gradients_from_a_frozen_base(make_model: Maker) -> None:
    model = make_model()
    model.requires_grad_(False)
    network = MOSSImageLoRANetwork(model, rank=2, targets="all")
    _train(network)

    network(torch.randn(1, 128, 2, 2), torch.full((1,), 0.4), torch.randn(1, 3, 8)).square().sum().backward()

    assert all(p.grad is not None for p in network.trainable_parameters())
    assert all(p.grad is None for p in model.parameters())
    assert len(list(network.trainable_parameters())) == 2 * len(network.lora_layers)
