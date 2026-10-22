from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from maltrainer.models.supra2_img.lora import DEFAULT_SUPRA_TARGET_MODULES, SupraLoRALayer, SupraLoRANetwork
from maltrainer.models.supra2_img.model import SupraDiT

DEPTH = 2


def _dit() -> SupraDiT:
    torch.manual_seed(0)
    return SupraDiT(latent_ch=4, d_model=32, depth=DEPTH, n_heads=4, ctx_dim=16, num_tokens=16)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return torch.randn(2, 4, 8, 8, generator=generator), torch.rand(2, generator=generator), torch.randn(2, 5, 16)


def test_a_fresh_lora_layer_adds_nothing_because_its_up_projection_is_zero() -> None:
    layer = SupraLoRALayer(nn.Linear(6, 3), rank=2, alpha=4.0)

    out = layer(torch.randn(5, 6))

    assert torch.equal(out, torch.zeros(5, 3))
    assert layer.scale == 2.0 and layer.lora_down.weight.shape == (2, 6) and layer.lora_up.weight.shape == (3, 2)
    assert layer.lora_down.weight.detach().abs().sum() > 0


def test_the_lora_output_is_up_of_down_times_alpha_over_rank() -> None:
    torch.manual_seed(0)
    layer = SupraLoRALayer(nn.Linear(6, 3), rank=4, alpha=2.0)
    layer.lora_up.weight.data.normal_()
    x = torch.randn(5, 6)

    out = layer(x)

    expected = x @ layer.lora_down.weight.t() @ layer.lora_up.weight.t() * (2.0 / 4)
    torch.testing.assert_close(out, expected)


def test_every_default_target_in_every_block_gets_one_adapter() -> None:
    network = SupraLoRANetwork(_dit(), rank=4, alpha=8.0)

    assert len(network.lora_layers) == len(DEFAULT_SUPRA_TARGET_MODULES) * DEPTH
    assert network.name_map["blocks_1_cross_attn_kv"] == "blocks.1.cross_attn.kv"
    assert network.name_map["blocks_0_mlp_2"] == "blocks.0.mlp.2"
    assert not any(not name.startswith("blocks.") for name in network.name_map.values())
    assert not any("adaln" in name for name in network.name_map.values())


def test_a_freshly_injected_network_leaves_the_prediction_unchanged() -> None:
    dit, (z, t, ctx) = _dit(), _inputs()
    before = dit(z, t, ctx)

    network = SupraLoRANetwork(dit, rank=4, alpha=4.0)

    torch.testing.assert_close(network(z, t, ctx), before)
    torch.testing.assert_close(dit(z, t, ctx), before)


def test_a_wrapped_linear_adds_the_adapter_output_to_its_own() -> None:
    dit = _dit()
    network = SupraLoRANetwork(dit, rank=4, alpha=8.0, target_modules=["cross_attn.q"])
    target = dit.blocks[1].cross_attn.q
    layer = network.lora_layers["blocks_1_cross_attn_q"]
    layer.lora_up.weight.data.normal_()
    x = torch.randn(3, 32)

    out = target(x)

    torch.testing.assert_close(out, F.linear(x, target.weight, target.bias) + layer(x))
    assert not torch.allclose(out, F.linear(x, target.weight, target.bias))


def test_only_the_adapters_receive_gradients() -> None:
    dit, (z, t, ctx) = _dit(), _inputs()
    dit.requires_grad_(False)
    network = SupraLoRANetwork(dit, rank=2, alpha=2.0, target_modules=["mlp.0"])
    network.lora_layers["blocks_0_mlp_0"].lora_up.weight.data.normal_(std=0.1)

    network(z, t, ctx).square().mean().backward()

    trainable = list(network.trainable_parameters())
    assert len(trainable) == 2 * DEPTH
    assert all(p.grad is not None for p in trainable)
    assert all(p.grad is None for p in dit.parameters())


def test_adapters_can_be_moved_to_another_dtype_without_touching_the_base_model() -> None:
    dit = _dit()
    network = SupraLoRANetwork(dit, rank=2, alpha=2.0)

    network.set_lora_dtype(torch.device("cpu"), torch.bfloat16)

    assert {p.dtype for p in network.trainable_parameters()} == {torch.bfloat16}
    assert dit.blocks[0].mlp[0].weight.dtype == torch.float32


@pytest.mark.parametrize("targets", [["nonexistent"], ["x_embed"], ["ctx_proj"]])
def test_targets_matching_no_block_linear_are_an_error(targets: list[str]) -> None:
    with pytest.raises(RuntimeError, match="No SupraDiT linears matched LoRA targets"):
        SupraLoRANetwork(_dit(), target_modules=targets)
