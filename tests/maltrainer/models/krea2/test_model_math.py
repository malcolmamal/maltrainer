"""The small math pieces of the Krea 2 DiT, against hand computations."""

from __future__ import annotations

import math

import pytest
import torch

from maltrainer.models.krea2.model import (
    DoubleSharedModulation,
    RMSNorm,
    SimpleModulation,
    SwiGLU,
    apply_rope1,
    embed_nd,
    rope,
    timestep_embedding,
)


def test_timestep_embedding_is_cos_then_sin_of_the_thousandfold_time():
    out = timestep_embedding(torch.tensor([0.5]), 4)
    # t*1000 = 500; frequencies exp(-ln(1e4) * [0, 1] / 2) = [1, 0.01]
    expected = [math.cos(500), math.cos(5), math.sin(500), math.sin(5)]
    assert out.shape == (1, 4)
    assert out.dtype == torch.float32
    assert out[0].tolist() == pytest.approx(expected, abs=1e-3)


def test_timestep_embedding_at_zero_is_ones_then_zeros():
    out = timestep_embedding(torch.zeros(3), 8)
    assert out[:, :4].eq(1).all() and out[:, 4:].eq(0).all()


def test_rope_is_the_rotation_matrix_of_position_times_frequency():
    out = rope(torch.tensor([[2.0]]), 4, 100.0)  # frequencies 1 and 100 ** -0.5 = 0.1
    assert out.shape == (1, 1, 2, 2, 2)
    for freq, angle in enumerate((2.0, 0.2)):
        expected = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        assert torch.allclose(out[0, 0, freq], torch.tensor(expected), atol=1e-6)


def test_embed_nd_concatenates_the_axes_and_adds_a_head_dim():
    ids = torch.zeros(2, 5, 3)
    ids[..., 1] = 1.0
    out = embed_nd(ids, [4, 6, 6], 1e3)
    assert out.shape == (2, 1, 5, 8, 2, 2)  # 4/2 + 6/2 + 6/2 rotations per token
    assert out[0, 0, 0, :2].eq(torch.eye(2)).all()  # axis 0 sits at position 0: identity
    assert not out[0, 0, 0, 2:5].eq(torch.eye(2)).all()  # axis 1 sits at position 1


def test_apply_rope_turns_each_channel_pair_by_its_angle():
    angle = math.pi / 2
    matrix = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    x = torch.tensor([[[[3.0, 4.0]]]])  # (b, heads, seq, headdim)
    out = apply_rope1(x, matrix.reshape(1, 1, 1, 1, 2, 2))
    assert out.flatten().tolist() == pytest.approx([-4.0, 3.0], abs=1e-6)


def test_apply_rope_keeps_the_input_dtype_and_vector_length():
    x = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    freqs = embed_nd(torch.zeros(1, 3, 1), [4], 1e3)
    out = apply_rope1(x, freqs)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float().norm(dim=-1), x.float().norm(dim=-1), rtol=1e-2)


def test_rms_norm_divides_by_the_root_mean_square():
    out = RMSNorm(2, eps=0.0)(torch.tensor([[3.0, 4.0]]))
    assert out.flatten().tolist() == pytest.approx([3 / math.sqrt(12.5), 4 / math.sqrt(12.5)], rel=1e-5)


def test_rms_norm_scale_is_stored_zero_centred():
    norm = RMSNorm(2, eps=0.0)
    norm.scale.data.fill_(1.0)  # weight 1 + 1 = 2
    out = norm(torch.tensor([[3.0, 4.0]]))
    assert out.flatten().tolist() == pytest.approx([6 / math.sqrt(12.5), 8 / math.sqrt(12.5)], rel=1e-5)


def test_rms_norm_returns_the_input_dtype():
    assert RMSNorm(4)(torch.randn(2, 4, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_swiglu_hidden_size_is_two_thirds_times_multiplier_rounded_up_to_128():
    assert SwiGLU(64, 4).gate.out_features == 256  # int(128 / 3) * 4 = 168 -> 256
    assert SwiGLU(384, 4).gate.out_features == 1024  # 256 * 4 = 1024 exactly
    assert SwiGLU(6, 1, multiple=1).down.in_features == 4


def test_swiglu_is_down_of_silu_gate_times_up():
    mlp = SwiGLU(2, 1, multiple=1)  # hidden size 1
    mlp.gate.weight.data = torch.tensor([[1.0, 0.0]])
    mlp.up.weight.data = torch.tensor([[0.0, 1.0]])
    mlp.down.weight.data = torch.tensor([[2.0], [3.0]])
    out = mlp(torch.tensor([[1.0, 2.0]]))
    silu_one = 1 / (1 + math.exp(-1))
    assert out.flatten().tolist() == pytest.approx([2 * silu_one * 2, 3 * silu_one * 2], rel=1e-5)


def test_double_shared_modulation_adds_its_vector_then_splits_in_six():
    mod = DoubleSharedModulation(2)
    mod.lin.data = torch.arange(12, dtype=torch.float32)
    parts = mod(torch.ones(1, 1, 12))
    assert len(parts) == 6
    assert [p.flatten().tolist() for p in parts] == [[1.0 + 2 * i, 2.0 + 2 * i] for i in range(6)]


def test_simple_modulation_gives_scale_and_shift_rows():
    mod = SimpleModulation(3)
    mod.lin.data = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    scale, shift = mod(torch.zeros(2, 1, 3))
    assert scale.shape == shift.shape == (2, 1, 3)
    assert scale[0, 0].tolist() == [1.0, 2.0, 3.0]
    assert shift[1, 0].tolist() == [10.0, 20.0, 30.0]
