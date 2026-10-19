from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.qwen_image_2_1.lora import QwenImage21LoRANetwork

from ._tiny import tiny_transformer

NAME = "diffusion_model.transformer_blocks.0.attn.to_q"


def _network(**options: object) -> QwenImage21LoRANetwork:
    network = QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=4.0, target_modules=["attn.to_q"], **options)
    torch.nn.init.normal_(network.lora_layers["transformer_blocks_0_attn_to_q"].lora_up.weight)
    return network


def test_the_default_format_uses_prefixed_lora_a_and_b_names_plus_an_alpha() -> None:
    state = _network().get_state_dict()

    assert len(state) == 6
    assert {f"{NAME}.lora_A.weight", f"{NAME}.lora_B.weight", f"{NAME}.alpha"} <= set(state)
    assert state[f"{NAME}.lora_A.weight"].shape == (2, 16)
    assert state[f"{NAME}.lora_B.weight"].shape == (16, 2)


def test_the_kohya_format_uses_down_and_up_names() -> None:
    state = _network().get_state_dict(format_type="kohya")

    assert {f"{NAME}.lora_down.weight", f"{NAME}.lora_up.weight", f"{NAME}.alpha"} <= set(state)
    assert not any("lora_A" in key for key in state)


def test_the_prefix_is_configurable() -> None:
    state = _network(prefix="unet.").get_state_dict()

    assert "unet.transformer_blocks.1.attn.to_q.alpha" in state


def test_weights_are_saved_as_cpu_bfloat16_by_default_and_alpha_as_float32() -> None:
    state = _network().get_state_dict()

    assert state[f"{NAME}.lora_B.weight"].dtype == torch.bfloat16
    assert state[f"{NAME}.alpha"].dtype == torch.float32
    assert state[f"{NAME}.alpha"].item() == 4.0
    assert all(not value.requires_grad and value.device.type == "cpu" for value in state.values())


def test_a_none_save_dtype_keeps_the_training_precision_exactly() -> None:
    network = _network()
    layer = network.lora_layers["transformer_blocks_0_attn_to_q"]

    state = network.get_state_dict(save_dtype=None)

    assert state[f"{NAME}.lora_B.weight"].dtype == torch.float32
    assert torch.equal(state[f"{NAME}.lora_B.weight"], layer.lora_up.weight.detach())


@pytest.mark.parametrize("format_type", ["diffusers", "kohya"])
def test_a_saved_adapter_loads_back_into_a_fresh_network_with_identical_output(
    format_type: str, tmp_path: Path
) -> None:
    source = _network()
    path = tmp_path / "lora.safetensors"
    save_file(source.get_state_dict(save_dtype=None, format_type=format_type), str(path))
    target = QwenImage21LoRANetwork(tiny_transformer(), rank=2, alpha=4.0, target_modules=["attn.to_q"])
    x = torch.randn(3, 16, generator=torch.Generator().manual_seed(0))

    target.load_assistant_adapter(path)

    expected = source.transformer.transformer_blocks[0].attn.to_q(x)
    torch.testing.assert_close(target.transformer.transformer_blocks[0].attn.to_q(x), expected)
