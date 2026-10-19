from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from maltrainer.models.moss_image.text_encoder import (
    MLP,
    GatedAttention,
    GatedDeltaNet,
    Qwen35TextConfig,
    Qwen35TextModel,
    RMSNorm,
    RMSNormGated,
    chunk_gated_delta_rule,
    rotate_half,
)

from .tiny_text import TINY, tiny_text_model


def _reference_delta_rule(query, key, value, g, beta) -> torch.Tensor:
    """The gated delta rule one token at a time: decay the state, correct it toward v, read it with q."""
    query, key = F.normalize(query, dim=-1) / query.shape[-1] ** 0.5, F.normalize(key, dim=-1)
    B, L, H, Dk = key.shape
    state = torch.zeros(B, H, Dk, value.shape[-1])
    out = []
    for t in range(L):
        state = state * g[:, t, :, None, None].exp()
        error = (value[:, t] - (state * key[:, t, :, :, None]).sum(-2)) * beta[:, t, :, None]
        state = state + key[:, t, :, :, None] * error[:, :, None, :]
        out.append((state * query[:, t, :, :, None]).sum(-2))
    return torch.stack(out, dim=1)


@pytest.mark.parametrize("length", [5, 64, 70])
def test_the_chunked_delta_rule_equals_the_token_by_token_recurrence(length: int) -> None:
    torch.manual_seed(length)
    q, k = torch.randn(1, length, 2, 4), torch.randn(1, length, 2, 4)
    v, g, beta = torch.randn(1, length, 2, 3), -torch.rand(1, length, 2), torch.rand(1, length, 2)

    out = chunk_gated_delta_rule(q, k, v, g=g, beta=beta)

    assert out.shape == (1, length, 2, 3)
    torch.testing.assert_close(out, _reference_delta_rule(q, k, v, g, beta), atol=1e-4, rtol=1e-4)


def test_rotate_half_swaps_the_halves_and_negates_the_first() -> None:
    assert rotate_half(torch.tensor([1.0, 2.0, 3.0, 4.0])).tolist() == [-3.0, -4.0, 1.0, 2.0]


def test_a_plain_norm_scales_by_the_weight_and_an_additive_one_by_one_plus_the_weight() -> None:
    x = torch.tensor([[3.0, 4.0]])
    plain, additive = RMSNorm(2, 0.0, add=False), RMSNorm(2, 0.0, add=True)
    plain.weight.data = additive.weight.data = torch.tensor([1.0, 3.0])

    unit = x / (12.5**0.5)
    torch.testing.assert_close(plain(x), unit * torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(additive(x), unit * torch.tensor([2.0, 4.0]))
    assert RMSNorm(2, 0.0, add=True).weight.tolist() == [0.0, 0.0]


def test_a_gated_norm_multiplies_the_normalised_input_by_silu_of_the_gate() -> None:
    norm = RMSNormGated(2, eps=0.0)
    x, gate = torch.tensor([[3.0, 4.0]]), torch.tensor([[1.0, -2.0]])

    torch.testing.assert_close(norm(x, gate), x / (12.5**0.5) * F.silu(gate))


def test_the_mlp_is_a_silu_gated_projection() -> None:
    torch.manual_seed(0)
    mlp, x = MLP(4, 6), torch.randn(2, 4)

    expected = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))

    torch.testing.assert_close(mlp(x), expected)


def test_linear_attention_never_reads_ahead() -> None:
    torch.manual_seed(1)
    layer, x = GatedDeltaNet(TINY), torch.randn(1, 9, 16)
    changed = x.clone()
    changed[:, 5:] += 1.0

    same, different = layer(x), layer(changed)

    torch.testing.assert_close(same[:, :5], different[:, :5], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(same[:, 5:], different[:, 5:])


def test_every_fourth_layer_is_full_attention_and_the_others_are_linear() -> None:
    model = Qwen35TextModel(TINY)

    kinds = ["self_attn" if hasattr(layer, "self_attn") else "linear_attn" for layer in model.layers]

    assert kinds == ["linear_attn"] * 3 + ["self_attn"]
    assert Qwen35TextConfig().full_attention_layers == tuple(range(3, 32, 4))


def test_the_text_model_is_causal_so_a_prefix_encodes_the_same_alone_or_in_a_longer_prompt() -> None:
    model = tiny_text_model()
    ids = torch.tensor([[3, 7, 1, 9, 4, 2, 8]])

    with torch.no_grad():
        full, prefix = model(ids), model(ids[:, :4])

    assert full.shape == (1, 7, 16)
    torch.testing.assert_close(full[:, :4], prefix, atol=1e-5, rtol=1e-5)


def test_the_text_rope_rotates_the_first_quarter_of_each_head_and_starts_unrotated() -> None:
    cos, sin = Qwen35TextModel(TINY).rope(3, torch.device("cpu"))

    assert cos.shape == sin.shape == (1, 1, 3, 2)
    assert cos[0, 0, 0].tolist() == [1.0, 1.0]
    assert sin[0, 0, 0].tolist() == [0.0, 0.0]
    torch.testing.assert_close(cos[0, 0, 2, 0], torch.tensor(float(torch.cos(torch.tensor(2.0)))))


def test_the_config_is_read_back_from_the_weights_it_built() -> None:
    state = dict(Qwen35TextModel(TINY).state_dict())

    assert Qwen35TextConfig.from_state_dict(state) == TINY


def test_the_default_config_is_what_the_9b_checkpoint_reads_back_as() -> None:
    with torch.device("meta"):
        state = Qwen35TextModel(Qwen35TextConfig()).state_dict()

    assert Qwen35TextConfig.from_state_dict(state) == Qwen35TextConfig()


def test_the_config_rejects_weights_that_are_not_qwen35() -> None:
    with pytest.raises(ValueError, match="Not a Qwen3.5 text encoder"):
        Qwen35TextConfig.from_state_dict({"layers.0.mlp.gate_proj.weight": torch.zeros(1, 1)})


def test_linear_attention_with_as_many_value_heads_as_key_heads_needs_no_repetition() -> None:
    torch.manual_seed(2)
    layer = GatedDeltaNet(replace(TINY, linear_num_key_heads=2))

    out = layer(torch.randn(1, 6, 16))

    assert out.shape == (1, 6, 16)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_full_attention_is_causal_with_or_without_grouped_key_value_heads(kv_heads: int) -> None:
    torch.manual_seed(3)
    layer = GatedAttention(replace(TINY, num_key_value_heads=kv_heads))
    cos, sin = Qwen35TextModel(TINY).rope(8, torch.device("cpu"))
    x = torch.randn(1, 8, 16)
    changed = x.clone()
    changed[:, 5:] += 1.0

    same, different = layer(x, cos, sin), layer(changed, cos, sin)

    torch.testing.assert_close(same[:, :5], different[:, :5], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(same[:, 5:], different[:, 5:])
