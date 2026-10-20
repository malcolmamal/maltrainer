from __future__ import annotations

import torch

from maltrainer.models.krea2.model import (
    Attention,
    DoubleSharedModulation,
    LastLayer,
    SimpleModulation,
    SingleStreamBlock,
    TextFusionBlock,
    TextFusionTransformer,
    embed_nd,
)

D = 32


def _attention(kvheads: int | None, bias: bool = False, seed: int = 0) -> Attention:
    torch.manual_seed(seed)
    return Attention(D, heads=2, kvheads=kvheads, bias=bias).eval()


def _freqs(batch: int, length: int) -> torch.Tensor:
    ids = torch.zeros(batch, length, 3)
    ids[..., 1] = torch.arange(length, dtype=torch.float32)
    return embed_nd(ids, [4, 6, 6], theta=1000.0)


def test_grouped_query_attention_equals_full_attention_with_the_shared_projections_repeated() -> None:
    grouped, full = _attention(kvheads=1), _attention(kvheads=None, seed=1)
    state = dict(grouped.state_dict())
    state["wk.weight"] = grouped.wk.weight.repeat(2, 1)
    state["wv.weight"] = grouped.wv.weight.repeat(2, 1)
    full.load_state_dict(state)
    x = torch.randn(2, 5, D)

    torch.testing.assert_close(grouped(x, _freqs(2, 5)), full(x, _freqs(2, 5)), atol=1e-5, rtol=1e-5)


def test_the_projections_have_the_head_counts_of_the_config() -> None:
    attention = _attention(kvheads=1)

    assert attention.headdim == 16
    assert (attention.wq.out_features, attention.wk.out_features, attention.wv.out_features) == (32, 16, 16)
    assert _attention(kvheads=None).wk.out_features == 32


def test_rope_changes_the_attention_output_and_an_identity_rope_does_not() -> None:
    attention = _attention(kvheads=1)
    x = torch.randn(1, 4, D)
    still = _freqs(1, 4).clone()
    still[:] = torch.eye(2)

    assert torch.allclose(attention(x), attention(x, still), atol=1e-6)
    assert not torch.allclose(attention(x), attention(x, _freqs(1, 4)))


def test_a_mask_keeps_a_token_from_attending_to_the_keys_it_hides() -> None:
    attention = _attention(kvheads=1)
    x = torch.randn(1, 3, D)
    open_mask = torch.ones(3, 3, dtype=torch.bool)
    hidden = open_mask.clone()
    hidden[:, 2] = False
    changed = x.clone()
    changed[:, 2] += 3.0

    assert torch.allclose(attention(x, mask=open_mask), attention(x), atol=1e-6)
    torch.testing.assert_close(attention(changed, mask=hidden)[:, :2], attention(x, mask=hidden)[:, :2])


def test_the_output_gate_is_a_sigmoid_of_a_projection_of_the_input() -> None:
    attention = _attention(kvheads=1, bias=True)
    attention.gate.weight.data.zero_()
    attention.wo.bias.data.zero_()
    x = torch.randn(1, 3, D)
    attention.gate.bias.data.fill_(60.0)
    open_gate = attention(x)

    attention.gate.bias.data.zero_()

    torch.testing.assert_close(attention(x), 0.5 * open_gate, atol=1e-5, rtol=1e-4)

    attention.gate.bias.data.fill_(-60.0)

    torch.testing.assert_close(attention(x), torch.zeros_like(open_gate), atol=1e-5, rtol=1e-4)


def test_the_double_modulation_adds_its_offsets_to_the_vector_and_splits_it_in_six() -> None:
    modulation = DoubleSharedModulation(4)
    modulation.lin.data = torch.arange(24.0)
    vec = torch.ones(2, 1, 24)

    chunks = modulation(vec)

    assert len(chunks) == 6 and all(c.shape == (2, 1, 4) for c in chunks)
    assert chunks[2][0, 0].tolist() == [9.0, 10.0, 11.0, 12.0]


def test_the_simple_modulation_gives_a_scale_and_a_shift_from_its_two_offset_rows() -> None:
    modulation = SimpleModulation(3)
    modulation.lin.data = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    scale, shift = modulation(torch.zeros(2, 1, 3))

    assert scale.shape == shift.shape == (2, 1, 3)
    assert scale[0, 0].tolist() == [1.0] * 3 and shift[0, 0].tolist() == [2.0] * 3


def test_a_single_stream_block_with_closed_gates_passes_its_input_through() -> None:
    torch.manual_seed(0)
    block = SingleStreamBlock(D, 2, 1, kvheads=1).eval()
    x = torch.randn(2, 4, D)

    out = block(x, torch.zeros(2, 1, 6 * D), _freqs(2, 4))

    assert torch.equal(out, x)


def test_opening_only_the_post_gate_adds_the_mlp_of_the_normalised_input() -> None:
    torch.manual_seed(0)
    block = SingleStreamBlock(D, 2, 1, kvheads=1).eval()
    x = torch.randn(2, 4, D)
    vec = torch.zeros(2, 1, 6 * D)
    vec[..., 5 * D :] = 1.0

    out = block(x, vec, _freqs(2, 4))

    torch.testing.assert_close(out, x + block.mlp(block.postnorm(x)))


def test_the_text_fusion_collapses_the_layer_axis_into_one_vector_per_token() -> None:
    torch.manual_seed(0)
    fusion = TextFusionTransformer(3, D, 2, 1, kvheads=1).eval()
    x = torch.randn(2, 5, 3, D)

    out = fusion(x)

    assert out.shape == (2, 5, D)
    assert fusion.projector.weight.shape == (1, 3)
    mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    assert not torch.allclose(fusion(x, mask=mask), out)


def test_a_text_fusion_block_is_a_pre_norm_residual_attention_and_mlp() -> None:
    torch.manual_seed(0)
    block = TextFusionBlock(D, 2, 1, kvheads=1).eval()
    block.attn.wo.weight.data.zero_()
    block.mlp.down.weight.data.zero_()
    x = torch.randn(1, 3, D)

    assert torch.equal(block(x), x)


def test_the_last_layer_projects_the_modulated_norm_to_patch_pixels() -> None:
    torch.manual_seed(0)
    layer = LastLayer(D, patch=2, channels=4)
    x = torch.randn(2, 3, D)

    out = layer(x, torch.zeros(2, 1, D))

    assert out.shape == (2, 3, 16)
    torch.testing.assert_close(out, layer.linear(layer.norm(x)))
