from __future__ import annotations

import math

import pytest
import torch

from maltrainer.models.ltx25.config import ConnectorConfig
from maltrainer.models.ltx25.connector import Embeddings1DConnector
from maltrainer.models.ltx25.layers import (
    AdaLayerNormSingle,
    CrossAttention,
    FeedForward,
    RMSNorm,
    ada_values,
    plain_rms,
    timestep_embedding,
)
from maltrainer.models.ltx25.rope import make_rope

from .tiny_model import randomise


def test_plain_rms_gives_unit_root_mean_square() -> None:
    x = torch.randn(2, 5, 8) * 3

    assert plain_rms(x).pow(2).mean(-1).sqrt().flatten().tolist() == pytest.approx([1.0] * 10, abs=1e-3)


def test_the_rms_norm_scales_by_its_weight_in_the_input_dtype() -> None:
    norm = RMSNorm(4, 1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    out = norm(torch.ones(1, 4).bfloat16())

    assert out.dtype == torch.bfloat16 and out.float().tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_the_timestep_embedding_is_cosines_then_sines_from_frequency_one_down() -> None:
    emb = timestep_embedding(torch.tensor([0.0, 2.0]))

    assert emb.shape == (2, 256)
    assert emb[0, :128].tolist() == [1.0] * 128 and emb[0, 128:].tolist() == [0.0] * 128
    assert emb[1, 0].item() == pytest.approx(math.cos(2.0)) and emb[1, 128].item() == pytest.approx(math.sin(2.0))
    assert emb[1, 127].item() == pytest.approx(math.cos(2.0 * math.exp(-math.log(10000) * 127 / 128)), abs=1e-5)


def test_ada_layer_norm_single_returns_the_modulation_and_the_embedding_it_came_from() -> None:
    adaln = AdaLayerNormSingle(16, 6)
    randomise(adaln)

    modulation, embedded = adaln(torch.tensor([0.0, 500.0]), torch.float32)

    assert modulation.shape == (2, 96) and embedded.shape == (2, 16)
    assert not torch.allclose(modulation[0], modulation[1])


def test_ada_values_add_the_table_row_to_the_matching_slice_of_the_timestep() -> None:
    table = torch.arange(6.0).view(3, 2)  # three rows of two
    timestep = torch.zeros(1, 4, 6)
    timestep[0, :, 2:4] = 10.0  # the second row

    first, second, third = ada_values(table, timestep)
    (only,) = ada_values(table, timestep, slice(1, 2))

    assert first[0, 0].tolist() == [0.0, 1.0] and third[0, 3].tolist() == [4.0, 5.0]
    assert second[0, 0].tolist() == [12.0, 13.0] and torch.equal(only, second)


def test_a_feed_forward_is_gelu_tanh_then_a_linear() -> None:
    ffn = FeedForward(4, True)
    randomise(ffn)
    x = torch.randn(2, 3, 4)

    expected = ffn.get_submodule("net.2")(
        torch.nn.functional.gelu(ffn.get_submodule("net.0.proj")(x), approximate="tanh")
    )

    assert torch.allclose(ffn(x), expected)
    assert ffn.get_parameter("net.0.proj.weight").shape == (16, 4)


def test_a_zero_gate_logit_leaves_attention_unchanged_because_the_gate_is_two_sigmoid() -> None:
    gated = CrossAttention(16, None, 2, 8, True)
    plain = CrossAttention(16, None, 2, 8, False)
    randomise(gated)
    plain.load_state_dict({k: v for k, v in gated.state_dict().items() if "gate" not in k})
    with torch.no_grad():
        gated.get_parameter("to_gate_logits.weight").zero_()
        gated.get_parameter("to_gate_logits.bias").zero_()
    x = torch.randn(1, 5, 16)

    assert torch.allclose(gated(x), plain(x), atol=1e-6)


def test_a_gate_logit_scales_the_heads_between_zero_and_two() -> None:
    gated, plain = CrossAttention(16, None, 2, 8, True), CrossAttention(16, None, 2, 8, False)
    randomise(gated)
    with torch.no_grad():
        gated.get_parameter("to_out.0.bias").zero_()
        gated.get_parameter("to_gate_logits.weight").zero_()
    plain.load_state_dict({k: v for k, v in gated.state_dict().items() if "gate" not in k})
    x = torch.randn(1, 5, 16)

    with torch.no_grad():
        gated.get_parameter("to_gate_logits.bias").fill_(100.0)
        doubled = gated(x)
        gated.get_parameter("to_gate_logits.bias").fill_(-100.0)
        shut = gated(x)

    assert torch.allclose(doubled, 2 * plain(x), atol=1e-5)
    assert shut.abs().max().item() < 1e-5


def test_attention_to_a_context_uses_its_own_width_and_rope_changes_the_result() -> None:
    attn = CrossAttention(16, 24, 2, 8, False)
    randomise(attn)
    x, context = torch.randn(1, 4, 16), torch.randn(1, 4, 24)
    pe = make_rope(torch.tensor([[[0.0, 3.0, 7.0, 15.0]]]), 16, 2, (20,), 10000.0, True, True, False)

    assert attn.to_k.weight.shape == (16, 24)
    assert attn(x, context).shape == (1, 4, 16)
    assert not torch.allclose(attn(x, context), attn(x, context, pe=pe))


CONNECTOR = ConnectorConfig(num_heads=2, head_dim=16, num_layers=1, registers=8, gated=True)


@pytest.mark.parametrize(("tokens", "expected"), [(1, 1024), (5, 1024), (1024, 1024), (1030, 1032), (2000, 2000)])
def test_the_connector_pads_with_register_tiles_up_to_a_thousand_tokens_or_the_next_whole_tile(
    tokens: int, expected: int
) -> None:
    connector = Embeddings1DConnector(CONNECTOR)
    randomise(connector)

    out = connector(torch.randn(1, tokens, 32))

    assert out.shape == (1, expected, 32)


def test_the_connector_output_is_normalised_and_the_registers_fill_the_tail() -> None:
    connector = Embeddings1DConnector(CONNECTOR)
    randomise(connector)
    short, longer = torch.randn(1, 3, 32), torch.randn(1, 4, 32)

    out_short, out_longer = connector(short), connector(longer)

    assert out_short.pow(2).mean(-1).sqrt().mean().item() == pytest.approx(1.0, abs=1e-3)
    assert not torch.allclose(out_short[:, :3], out_longer[:, :3])  # the text attends to what follows it
    assert out_short.shape == out_longer.shape
