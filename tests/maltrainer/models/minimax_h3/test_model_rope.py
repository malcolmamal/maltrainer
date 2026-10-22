from __future__ import annotations

import math

import pytest
import torch

from maltrainer.models.minimax_h3.model import MiniMaxH3Attention, MiniMaxH3RotaryPosEmbed, _apply_rotary_emb


def _rope(position_ids: list[list[float]], freq_dim: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    return MiniMaxH3RotaryPosEmbed(freq_dim, 10000.0)(torch.tensor(position_ids))


def _attention() -> MiniMaxH3Attention:
    torch.manual_seed(0)
    return MiniMaxH3Attention(hidden_size=16, heads=2, head_dim=8, qk_norm_eps=1e-5)


def test_inverse_frequencies_fall_geometrically_from_one() -> None:
    rope = MiniMaxH3RotaryPosEmbed(2, 10000.0)

    torch.testing.assert_close(rope.inv_freq, torch.tensor([1.0, 0.01]))
    assert "inv_freq" in rope.state_dict()


def test_each_axis_contributes_its_own_block_of_angles_and_the_block_is_repeated() -> None:
    cos, sin = _rope([[1.0, 2.0, 3.0]])

    angles = torch.tensor([1.0, 0.01, 2.0, 0.02, 3.0, 0.03]).repeat(2)
    torch.testing.assert_close(cos, angles.cos()[None])
    torch.testing.assert_close(sin, angles.sin()[None])


def test_the_time_axis_only_rotates_the_first_block() -> None:
    cos, sin = _rope([[5.0, 0.0, 0.0]])

    torch.testing.assert_close(cos[0, 2:6], torch.ones(4))
    torch.testing.assert_close(sin[0, 2:6], torch.zeros(4))
    assert cos[0, 0].item() == pytest.approx(math.cos(5.0))


def test_position_zero_is_not_rotated() -> None:
    cos, sin = _rope([[0.0, 0.0, 0.0]])

    torch.testing.assert_close(cos, torch.ones(1, 12))
    torch.testing.assert_close(sin, torch.zeros(1, 12))


def test_a_rotation_by_a_quarter_turn_sends_the_first_half_to_the_second_and_negates_the_other() -> None:
    hidden = torch.arange(1.0, 9.0).view(1, 1, 1, 8)
    cos, sin = torch.zeros(1, 6), torch.ones(1, 6)

    out = _apply_rotary_emb(hidden, cos, sin)

    torch.testing.assert_close(out.flatten(), torch.tensor([-4.0, -5.0, -6.0, 1.0, 2.0, 3.0, 7.0, 8.0]))


def test_no_rotation_leaves_the_vector_alone() -> None:
    hidden = torch.randn(2, 3, 2, 8)

    out = _apply_rotary_emb(hidden, torch.ones(3, 6), torch.zeros(3, 6))

    torch.testing.assert_close(out, hidden)


def test_each_token_is_rotated_by_its_own_angles() -> None:
    hidden = torch.ones(1, 2, 1, 4)
    cos, sin = torch.tensor([[1.0] * 4, [0.0] * 4]), torch.tensor([[0.0] * 4, [1.0] * 4])

    out = _apply_rotary_emb(hidden, cos, sin)

    torch.testing.assert_close(out[0, :, 0], torch.tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0]]))


def test_the_rotation_keeps_the_dtype_of_the_activations() -> None:
    out = _apply_rotary_emb(torch.ones(1, 1, 1, 8, dtype=torch.bfloat16), torch.ones(1, 6), torch.zeros(1, 6))

    assert out.dtype == torch.bfloat16


def test_shifting_every_position_by_the_same_amount_leaves_attention_unchanged() -> None:
    attention = _attention()
    x = torch.randn(1, 5, 16)
    positions = torch.tensor([[0.0, 0, 0], [1, 0, 2], [2, 1, 0], [3, 1, 1], [4, 2, 3]])
    rope = MiniMaxH3RotaryPosEmbed(1, 10000.0)

    base = attention(x, rope(positions))
    shifted = attention(x, rope(positions + torch.tensor([7.0, 3.0, 5.0])))

    torch.testing.assert_close(shifted, base, atol=1e-4, rtol=1e-4)


def test_moving_a_single_token_changes_what_attention_returns() -> None:
    attention = _attention()
    x = torch.randn(1, 4, 16)
    positions = torch.tensor([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    moved = positions.clone()
    moved[3, 0] = 30.0
    rope = MiniMaxH3RotaryPosEmbed(1, 10000.0)

    base, other = attention(x, rope(positions)), attention(x, rope(moved))

    assert not torch.allclose(base, other, atol=1e-3)


def test_attention_without_rotary_embeddings_equals_attention_with_a_zero_angle() -> None:
    attention = _attention()
    x = torch.randn(1, 4, 16)
    rope = MiniMaxH3RotaryPosEmbed(1, 10000.0)

    plain = attention(x)
    unrotated = attention(x, rope(torch.zeros(4, 3)))

    torch.testing.assert_close(plain, unrotated)


def test_a_diagonal_mask_makes_each_token_see_only_its_own_value() -> None:
    attention = _attention()
    x = torch.randn(1, 4, 16)

    out = attention(x, attention_mask=torch.eye(4, dtype=torch.bool))

    value = attention.qkv_proj(x).chunk(3, dim=-1)[2]
    torch.testing.assert_close(out, attention.out_proj(value), atol=1e-6, rtol=1e-5)


def test_an_additive_mask_removes_attention_to_a_blocked_token() -> None:
    attention = _attention()
    x = torch.randn(1, 3, 16)
    blocked = torch.zeros(3, 3)
    blocked[:, 2] = float("-inf")
    changed = x.clone()
    changed[0, 2] += 5.0

    before = attention(x, attention_mask=blocked)[:, :2]
    after = attention(changed, attention_mask=blocked)[:, :2]

    torch.testing.assert_close(before, after)
