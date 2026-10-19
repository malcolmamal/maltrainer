from __future__ import annotations

import math

import pytest
import torch

from maltrainer.models.moss_image.model import (
    RMSNorm,
    apply_rope,
    attention,
    position_ids,
    rope,
    rope_from_ids,
    situ,
    timestep_embedding,
)


def test_timestep_embedding_is_cos_then_sin_of_1000_sigma_times_each_frequency() -> None:
    out = timestep_embedding(torch.tensor([0.0, 0.001]), dim=4)

    expected = torch.tensor([[1.0, 1.0, 0.0, 0.0], [math.cos(1), math.cos(0.01), math.sin(1), math.sin(0.01)]])
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0)


def test_timestep_embedding_defaults_to_256_features_whatever_the_sigma_dtype() -> None:
    out = timestep_embedding(torch.tensor([0.5, 0.25], dtype=torch.bfloat16))

    assert out.shape == (2, 256)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out[0, 1], torch.tensor(math.cos(500.0 * 10000 ** (-1 / 128))).float())


def test_rope_is_the_identity_at_position_zero_and_a_rotation_by_pos_times_omega_after() -> None:
    out = rope(torch.tensor([0.0, 1.0]), dim=4, theta=10000)

    assert out.shape == (2, 2, 2, 2)
    torch.testing.assert_close(out[0], torch.eye(2).expand(2, 2, 2))
    c1, s1, c2, s2 = math.cos(1), math.sin(1), math.cos(0.01), math.sin(0.01)
    torch.testing.assert_close(out[1, 0], torch.tensor([[c1, -s1], [s1, c1]]).float())
    torch.testing.assert_close(out[1, 1], torch.tensor([[c2, -s2], [s2, c2]]).float())


def test_position_ids_put_the_image_first_at_the_text_frame_with_centred_rows_and_columns() -> None:
    ids = position_ids(2, 3, 2, torch.device("cpu"))

    assert ids.shape == (8, 4)
    assert ids[:, 0].eq(0).all()
    assert ids[:6, 1].tolist() == [2.0] * 6
    assert ids[:6, 2].tolist() == [-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]
    assert ids[:6, 3].tolist() == [-1.0, 0.0, 1.0] * 2
    assert ids[6:, 1].tolist() == [0.0, 1.0]
    assert ids[6:, 2:].eq(0).all()


def test_rope_from_ids_gives_every_axis_a_quarter_of_the_head_and_leaves_axis_r_unrotated() -> None:
    ids = position_ids(2, 2, 3, torch.device("cpu"))

    pe = rope_from_ids(ids, head_dim=16)

    assert pe.shape == (1, 7, 1, 8, 2, 2)
    torch.testing.assert_close(pe[0, :, 0, :2], torch.eye(2).expand(7, 2, 2, 2))
    assert not torch.allclose(pe[0, :, 0, 2:4], torch.eye(2).expand(7, 2, 2, 2))


def test_apply_rope_rotates_the_pair_x_i_and_x_i_plus_half_and_keeps_the_dtype() -> None:
    quarter_turn = torch.tensor([[0.0, -1.0], [1.0, 0.0]]).expand(1, 1, 1, 2, 2, 2)
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.bfloat16)

    rotated = apply_rope(x, quarter_turn)
    unchanged = apply_rope(x, torch.eye(2).expand(1, 1, 1, 2, 2, 2))

    assert rotated.dtype == torch.bfloat16
    assert rotated.flatten().tolist() == [-3.0, -4.0, 1.0, 2.0]
    assert unchanged.flatten().tolist() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize(("hidden", "gate"), [(0.0, 3.0), (7.0, 0.0), (-1.5, 2.0), (30.0, -4.0)])
def test_situ_is_a_soft_capped_gate_times_a_soft_capped_value(hidden: float, gate: float) -> None:
    out = situ(torch.tensor(hidden), torch.tensor(gate))

    expected = 4 * math.tanh(gate / 4) / (1 + math.exp(-gate)) * 25 * math.tanh(hidden / 25)
    torch.testing.assert_close(out, torch.tensor(expected), atol=1e-5, rtol=1e-5)


def test_situ_saturates_at_a_hundred() -> None:
    out = situ(torch.tensor(1e6), torch.tensor(1e6))

    assert out.item() == pytest.approx(100.0)


def test_rms_norm_scales_by_the_root_mean_square_and_the_weight_in_the_inputs_dtype() -> None:
    norm = RMSNorm(4, eps=0.0)
    norm.weight.data = torch.tensor([1.0, 2.0, 1.0, 2.0])
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0], [3.0, 0.0, 0.0, 4.0]], dtype=torch.bfloat16)

    out = norm(x)

    rms = torch.tensor([[1.0], [math.sqrt(25 / 4)]])
    expected = x.float() / rms * norm.weight
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


def test_attention_with_zeroed_queries_and_keys_averages_the_values_over_the_sequence() -> None:
    B, N, heads, dim = 2, 5, 2, 16
    pe = rope_from_ids(position_ids(1, 3, 2, torch.device("cpu")), head_dim=dim // heads)
    v = torch.randn(B, N, dim)
    norms = RMSNorm(dim // heads), RMSNorm(dim // heads)

    out = attention(torch.zeros(B, N, dim), torch.zeros(B, N, dim), v, *norms, pe, heads)

    assert out.shape == (B, N, dim)
    expected = v.view(B, N, heads, -1).mean(dim=1, keepdim=True).expand(B, N, heads, dim // heads).reshape(B, N, dim)
    torch.testing.assert_close(out, expected)
