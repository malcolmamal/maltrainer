from __future__ import annotations

import pytest
import torch

from maltrainer.models.qwen_image_2_1.model import AdaLayerNormContinuous, Attention, TransformerBlock

MASK = torch.tensor([False, True])


def _attention() -> Attention:
    torch.manual_seed(0)
    return Attention(dim=8, heads=2, dim_head=4)


def test_attention_returns_a_tensor_shaped_like_its_input() -> None:
    out = _attention()(torch.randn(2, 5, 8))

    assert out.shape == (2, 5, 8)


def test_attention_projections_have_no_bias_and_the_qk_norms_are_per_head_dim() -> None:
    attention = _attention()

    assert sorted(attention.state_dict()) == [
        "norm_k.weight",
        "norm_q.weight",
        "to_k.weight",
        "to_out.0.weight",
        "to_q.weight",
        "to_v.weight",
    ]
    assert attention.norm_q.weight.shape == (4,)


def test_a_causal_mask_keeps_early_tokens_independent_of_later_ones() -> None:
    attention = _attention()
    x = torch.randn(1, 4, 8)
    changed = x.clone()
    changed[:, 2:] = torch.randn(1, 2, 8)
    causal = torch.ones(4, 4, dtype=torch.bool).tril()

    masked = attention(x, attn_mask=causal)[:, :2], attention(changed, attn_mask=causal)[:, :2]
    unmasked = attention(x)[:, :2], attention(changed)[:, :2]

    torch.testing.assert_close(masked[0], masked[1])
    assert not torch.allclose(unmasked[0], unmasked[1])


def test_rotary_embeddings_make_attention_depend_on_token_positions() -> None:
    attention = _attention()
    x = torch.randn(1, 3, 8)
    angles = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 0.5]])

    plain = attention(x)
    rotated = attention(x, rotary_emb=torch.polar(torch.ones(3, 2), angles))
    unit = attention(x, rotary_emb=torch.ones(3, 2, dtype=torch.complex64))

    torch.testing.assert_close(unit, plain)
    assert not torch.allclose(rotated, plain)


def test_modulation_without_a_token_mask_scales_and_gates_every_token_alike() -> None:
    block = TransformerBlock(dim=2, heads=1, dim_head=2)
    mod = torch.tensor([[1.0, 2.0, 0.5, -0.5]])

    out, gate = block._modulate(torch.ones(1, 3, 2), mod)

    torch.testing.assert_close(out, torch.tensor([[[2.0, 3.0]] * 3]))
    torch.testing.assert_close(gate, torch.tensor([[[0.5, -0.5]]]))


def test_modulation_with_a_token_mask_uses_the_last_row_for_the_unmasked_tokens() -> None:
    block = TransformerBlock(dim=2, heads=1, dim_head=2)
    mod = torch.tensor([[1.0, 1.0, 0.1, 0.2], [-0.5, 0.0, 0.3, 0.4]])

    out, gate = block._modulate(torch.ones(1, 2, 2), mod, MASK)

    torch.testing.assert_close(out, torch.tensor([[[0.5, 1.0], [2.0, 2.0]]]))
    torch.testing.assert_close(gate, torch.tensor([[[0.3, 0.4], [0.1, 0.2]]]))


def test_a_block_with_zero_modulation_is_the_identity() -> None:
    torch.manual_seed(0)
    block = TransformerBlock(dim=8, heads=2, dim_head=4, mlp_ratio=2)
    x = torch.randn(1, 3, 8)

    out = block(x, torch.zeros(1, 32))

    torch.testing.assert_close(out, x)


def test_a_block_adds_gated_attention_and_feed_forward_residuals() -> None:
    torch.manual_seed(0)
    block = TransformerBlock(dim=8, heads=2, dim_head=4, mlp_ratio=2)
    x, modulation = torch.randn(1, 3, 8), torch.randn(1, 32)
    mod1, mod2 = modulation.chunk(2, dim=-1)

    out = block(x, modulation)

    scaled, gate = block._modulate(block.img_norm1(x), mod1)
    mid = x + gate.tanh() * block.attn(scaled)
    scaled, gate = block._modulate(block.img_norm2(mid), mod2)
    torch.testing.assert_close(out, mid + gate.tanh() * block.img_mlp(scaled))


def test_the_final_norm_is_a_parameter_free_layer_norm_scaled_by_the_condition() -> None:
    norm = AdaLayerNormContinuous(embedding_dim=4, conditioning_embedding_dim=3)
    torch.nn.init.zeros_(norm.linear.weight)
    x = torch.randn(1, 2, 4)

    out = norm(x, torch.randn(1, 3))

    torch.testing.assert_close(out, torch.nn.functional.layer_norm(x, (4,), eps=1e-6))
    assert sorted(norm.state_dict()) == ["linear.weight"]


@pytest.mark.parametrize("masked", [False, True])
def test_the_final_norm_applies_the_silu_of_the_condition_as_a_scale_without_a_shift(masked: bool) -> None:
    norm = AdaLayerNormContinuous(embedding_dim=2, conditioning_embedding_dim=2)
    with torch.no_grad():
        norm.linear.weight.copy_(torch.eye(2))
    cond = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
    cond = cond if masked else cond[:1]
    x = torch.randn(1, 2, 2)

    out = norm(x, cond, MASK if masked else None)

    silu = torch.nn.functional.silu(cond)
    per_token = torch.stack([silu[1], silu[0]]) if masked else silu
    expected = torch.nn.functional.layer_norm(x, (2,), eps=1e-6) * (1.0 + per_token)
    torch.testing.assert_close(out, expected)
