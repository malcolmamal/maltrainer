from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRALayer, QwenImage21LoRANetwork

from ._tiny import tiny_transformer

Q0 = "transformer_blocks.0.attn.to_q"
Q1 = "transformer_blocks.1.attn.to_q"
DOWN, UP = torch.full((2, 16), 0.05), torch.full((16, 2), 0.5)


def _network(**options: object) -> QwenImage21LoRANetwork:
    return QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=2.0, target_modules=["attn.to_q"], **options)


def _adapter(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    path = tmp_path / "assistant.safetensors"
    save_file(tensors, str(path))
    return path


def _pair(name: str, alpha: float | None = 4.0, style: str = "diffusers") -> dict[str, torch.Tensor]:
    down, up = ("lora_A", "lora_B") if style == "diffusers" else ("lora_down", "lora_up")
    tensors = {f"{name}.{down}.weight": DOWN.clone(), f"{name}.{up}.weight": UP.clone()}
    if alpha is not None:
        tensors[f"{name}.alpha"] = torch.tensor(alpha)
    return tensors


def _delta(x: torch.Tensor, scale: float) -> torch.Tensor:
    return (x @ DOWN.T) @ UP.T * scale


def _output(network: QwenImage21LoRANetwork, name: str, x: torch.Tensor) -> torch.Tensor:
    return network.transformer.get_submodule(name)(x)


@pytest.mark.parametrize("style", ["diffusers", "kohya"])
def test_the_adapter_is_added_to_the_linear_output_in_either_naming_style(tmp_path: Path, style: str) -> None:
    network = _network()
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(0))
    base, other = _output(network, Q0, x), _output(network, Q1, x)

    attached = network.load_assistant_adapter(_adapter(tmp_path, _pair(f"diffusion_model.{Q0}", 4.0, style)))

    assert attached == 1
    torch.testing.assert_close(_output(network, Q0, x), base + _delta(x, 4.0 / 2))
    torch.testing.assert_close(_output(network, Q1, x), other)


def test_the_multiplier_scales_the_adapter_contribution(tmp_path: Path) -> None:
    network = _network()
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(1))
    base = _output(network, Q0, x)

    network.load_assistant_adapter(_adapter(tmp_path, _pair(Q0, 2.0)), multiplier=0.25)

    layer = network.assistant_layers["transformer_blocks_0_attn_to_q"]
    assert layer.scale == pytest.approx(0.25)
    torch.testing.assert_close(_output(network, Q0, x), base + _delta(x, 0.25))


def test_a_missing_alpha_means_alpha_equals_rank_so_the_scale_is_one(tmp_path: Path) -> None:
    network = _network()

    network.load_assistant_adapter(_adapter(tmp_path, _pair(Q0, alpha=None)))

    layer = network.assistant_layers["transformer_blocks_0_attn_to_q"]
    assert (layer.alpha, layer.rank, layer.scale) == (2.0, 2, 1.0)


def test_the_assistant_layers_are_frozen_in_eval_mode_and_outside_the_trainable_set(tmp_path: Path) -> None:
    network = _network()

    network.load_assistant_adapter(_adapter(tmp_path, {**_pair(Q0), **_pair(Q1)}))

    layers = list(network.assistant_layers.values())
    assert len(layers) == 2
    assert all(isinstance(layer, QwenImage21LoRALayer) and not layer.training for layer in layers)
    assert not any(p.requires_grad for layer in layers for p in layer.parameters())
    trainable = {id(p) for p in network.trainable_parameters()}
    assert not trainable & {id(p) for layer in layers for p in layer.parameters()}


def test_the_assistant_stacks_on_the_trainable_lora(tmp_path: Path) -> None:
    network = _network()
    own = network.lora_layers["transformer_blocks_0_attn_to_q"]
    torch.nn.init.constant_(own.lora_up.weight, 0.2)
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(2))
    before = _output(network, Q0, x)

    network.load_assistant_adapter(_adapter(tmp_path, _pair(Q0, 2.0)))

    torch.testing.assert_close(_output(network, Q0, x), before + _delta(x, 1.0))


def test_weights_can_be_swapped_in_place_and_a_zero_scale_restores_the_base(tmp_path: Path) -> None:
    network = _network()
    x = torch.randn(2, 16, generator=torch.Generator().manual_seed(3))
    base = _output(network, Q0, x)
    network.load_assistant_adapter(_adapter(tmp_path, _pair(Q0, 2.0)))
    layer = network.assistant_layers["transformer_blocks_0_attn_to_q"]

    layer.lora_up.weight.data.copy_(UP * 2)
    layer.scale = 0.5
    doubled = _output(network, Q0, x)
    layer.scale = 0.0

    torch.testing.assert_close(doubled, base + _delta(x, 1.0))
    torch.testing.assert_close(_output(network, Q0, x), base)


def test_keys_without_the_prefix_or_with_a_custom_prefix_are_resolved(tmp_path: Path) -> None:
    network = _network(prefix="unet.")

    attached = network.load_assistant_adapter(_adapter(tmp_path, {**_pair(f"unet.{Q0}"), **_pair(Q1)}))

    assert sorted(network.assistant_layers) == ["transformer_blocks_0_attn_to_q", "transformer_blocks_1_attn_to_q"]
    assert attached == 2


def test_modules_the_transformer_lacks_or_that_are_not_linears_are_skipped_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    network = _network()
    tensors = {**_pair(Q0), **_pair("transformer_blocks.0.nope"), **_pair("transformer_blocks.0.attn.to_out.1")}

    with caplog.at_level("WARNING", logger="maltrainer.models.qwen_image_2_1.lora"):
        attached = network.load_assistant_adapter(_adapter(tmp_path, tensors))

    assert attached == 1
    assert "skipped 2 module(s)" in caplog.text
    assert list(network.assistant_layers) == ["transformer_blocks_0_attn_to_q"]


def test_a_down_matrix_without_its_up_matrix_is_ignored(tmp_path: Path) -> None:
    network = _network()
    half = {f"{Q1}.lora_A.weight": DOWN.clone()}

    attached = network.load_assistant_adapter(_adapter(tmp_path, {**_pair(Q0), **half}))

    assert attached == 1
    assert list(network.assistant_layers) == ["transformer_blocks_0_attn_to_q"]


def test_the_adapter_can_be_moved_to_a_dtype_when_attached(tmp_path: Path) -> None:
    network = _network()

    network.load_assistant_adapter(_adapter(tmp_path, _pair(Q0)), dtype=torch.bfloat16, device=torch.device("cpu"))

    layer = network.assistant_layers["transformer_blocks_0_attn_to_q"]
    assert layer.lora_down.weight.dtype == layer.lora_up.weight.dtype == torch.bfloat16


def test_two_adapters_attach_in_one_assistant_collection(tmp_path: Path) -> None:
    network = _network()
    first = tmp_path / "a.safetensors"
    save_file(_pair(Q0), str(first))

    network.load_assistant_adapter(first)
    attached = network.load_assistant_adapter(_adapter(tmp_path, _pair(Q1)))

    assert attached == 1
    assert len(network.assistant_layers) == 2


def test_tensors_that_are_not_lora_weights_are_ignored(tmp_path: Path) -> None:
    network = _network()

    attached = network.load_assistant_adapter(_adapter(tmp_path, {**_pair(Q0), "metadata.scale": torch.ones(1)}))

    assert attached == 1
