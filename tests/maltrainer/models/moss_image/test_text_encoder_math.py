from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from maltrainer.models.moss_image.text_encoder import (
    MLP,
    GatedAttention,
    GatedDeltaNet,
    RMSNorm,
    RMSNormGated,
    chunk_gated_delta_rule,
    rotate_half,
)

from .qwen35_tiny import randomize, tiny_text_config, tiny_text_model

CPU = torch.device("cpu")


def _naive_delta_rule(q, k, v, g, beta) -> torch.Tensor:
    """The gated delta rule one token at a time: decay the state, correct it toward v, read with q."""
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    q = q / math.sqrt(q.shape[-1])
    batch, length, heads, dk = k.shape
    state = torch.zeros(batch, heads, dk, v.shape[-1], dtype=torch.float64)
    out = torch.zeros(batch, length, heads, v.shape[-1], dtype=torch.float64)
    for t in range(length):
        state = state * g[:, t].exp()[..., None, None]
        predicted = (state * k[:, t][..., None]).sum(-2)
        delta = (v[:, t] - predicted) * beta[:, t][..., None]
        state = state + k[:, t][..., None] * delta[..., None, :]
        out[:, t] = (state * q[:, t][..., None]).sum(-2)
    return out


@pytest.mark.parametrize("length", [1, 5, 64, 70])
def test_the_chunked_delta_rule_equals_the_token_by_token_recurrence(length: int) -> None:
    gen = torch.Generator().manual_seed(length)
    shape = (2, length, 2, 6)
    q, k, v = (torch.randn(*shape, generator=gen, dtype=torch.float64) for _ in range(3))
    g = -torch.rand(2, length, 2, generator=gen, dtype=torch.float64)
    beta = torch.rand(2, length, 2, generator=gen, dtype=torch.float64)

    got = chunk_gated_delta_rule(q, k, v, g, beta)

    assert got.shape == shape and got.dtype == torch.float64
    torch.testing.assert_close(got, _naive_delta_rule(q, k, v, g, beta), atol=1e-5, rtol=1e-4)


def test_the_delta_rule_returns_the_dtype_it_was_given_though_it_computes_in_fp32() -> None:
    x = torch.randn(1, 3, 1, 4, dtype=torch.bfloat16)

    out = chunk_gated_delta_rule(x, x, x, torch.zeros(1, 3, 1), torch.full((1, 3, 1), 0.5))

    assert out.dtype == torch.bfloat16


def test_rms_norm_scales_by_the_root_mean_square_and_one_plus_weight_when_additive() -> None:
    x = torch.tensor([[3.0, 4.0]])
    plain, additive = RMSNorm(2, 0.0, add=False), RMSNorm(2, 0.0, add=True)
    plain.weight.data = torch.tensor([2.0, 1.0])
    additive.weight.data = torch.tensor([1.0, 0.0])
    rms = math.sqrt((9 + 16) / 2)

    expected = torch.tensor([[3 / rms * 2, 4 / rms]])
    torch.testing.assert_close(plain(x), expected)
    torch.testing.assert_close(additive(x), expected)
    assert RMSNorm(2, 0.0, add=True).weight.abs().sum() == 0
    assert RMSNorm(2, 0.0, add=False).weight.sum() == 2


def test_the_gated_norm_multiplies_the_normalised_input_by_silu_of_the_gate() -> None:
    norm = RMSNormGated(3, eps=1e-6)
    x, gate = torch.randn(4, 3), torch.randn(4, 3)

    torch.testing.assert_close(norm(x, gate), F.rms_norm(x, (3,), eps=1e-6) * F.silu(gate))


def test_the_mlp_is_a_silu_gated_feed_forward() -> None:
    mlp = MLP(4, 6)
    x = torch.randn(2, 4)

    expected = (F.silu(x @ mlp.gate_proj.weight.T) * (x @ mlp.up_proj.weight.T)) @ mlp.down_proj.weight.T

    torch.testing.assert_close(mlp(x), expected)


def test_rotate_half_swaps_the_halves_and_negates_the_first() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert rotate_half(x).tolist() == [-3.0, -4.0, 1.0, 2.0]


def test_position_zero_is_not_rotated_and_later_positions_use_the_partial_rotary_width() -> None:
    model = tiny_text_model()

    cos, sin = model.rope(3, CPU)

    assert cos.shape == sin.shape == (1, 1, 3, 2)  # head_dim 8 * 0.25 = 2 rotary dims
    assert cos[0, 0, 0].tolist() == [1.0, 1.0] and sin[0, 0, 0].tolist() == [0.0, 0.0]
    assert cos[0, 0, 2, 0].item() == pytest.approx(math.cos(2.0))
    assert sin[0, 0, 2, 1].item() == pytest.approx(math.sin(2.0))


def _perturbed_tail(x: torch.Tensor) -> torch.Tensor:
    changed = x.clone()
    changed[:, 4:] += 5.0
    return changed


def test_gated_attention_is_causal_and_keeps_the_input_shape() -> None:
    config = tiny_text_config()
    attn = randomize(GatedAttention(config))
    cos, sin = tiny_text_model().rope(6, CPU)
    x = torch.randn(1, 6, config.hidden_size)

    out, out_changed = attn(x, cos, sin), attn(_perturbed_tail(x), cos, sin)

    assert out.shape == x.shape
    torch.testing.assert_close(out[:, :4], out_changed[:, :4])
    assert not torch.allclose(out[:, 4:], out_changed[:, 4:])


def test_gated_attention_scores_depend_on_the_relative_rotary_angle() -> None:
    config = tiny_text_config()
    attn = randomize(GatedAttention(config))
    x = torch.randn(1, 2, config.hidden_size)
    unrotated = (torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))
    rotated = tiny_text_model().rope(2, CPU)  # position 1 turns by one radian: relative angles change scores

    assert attn.rotary_dim == 2 and attn.q_proj.out_features == 2 * 2 * 8
    assert not torch.allclose(attn(x, *unrotated), attn(x, *rotated))


def test_gated_delta_net_is_causal_and_repeats_key_heads_across_value_heads() -> None:
    config = tiny_text_config()
    net = randomize(GatedDeltaNet(config))
    x = torch.randn(1, 6, config.hidden_size)

    out, out_changed = net(x), net(_perturbed_tail(x))

    assert out.shape == x.shape
    assert net.num_value_heads // net.num_key_heads == 2
    torch.testing.assert_close(out[:, :4], out_changed[:, :4])
    assert not torch.allclose(out[:, 4:], out_changed[:, 4:])


def test_attention_without_grouped_kv_heads_still_runs_causally() -> None:
    config = replace(tiny_text_config(), num_key_value_heads=2)
    attn = randomize(GatedAttention(config))
    cos, sin = tiny_text_model().rope(6, CPU)
    x = torch.randn(1, 6, config.hidden_size)

    out, out_changed = attn(x, cos, sin), attn(_perturbed_tail(x), cos, sin)

    torch.testing.assert_close(out[:, :4], out_changed[:, :4])


def test_delta_net_with_as_many_value_heads_as_key_heads_needs_no_repeat() -> None:
    config = replace(tiny_text_config(), linear_num_value_heads=1, linear_value_head_dim=8)
    net = randomize(GatedDeltaNet(config))
    x = torch.randn(1, 6, config.hidden_size)

    out = net(x)

    assert net.num_value_heads == net.num_key_heads == 1
    assert out.shape == x.shape
