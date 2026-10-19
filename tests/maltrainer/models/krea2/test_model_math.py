from __future__ import annotations

import math

import torch

from maltrainer.models.krea2.model import (
    RMSNorm,
    SwiGLU,
    apply_rope1,
    embed_nd,
    rope,
    timestep_embedding,
)


def test_the_timestep_embedding_is_cos_then_sin_of_a_thousand_times_sigma() -> None:
    out = timestep_embedding(torch.tensor([1.0, 0.0]), dim=4)

    expected = torch.tensor([[math.cos(1000.0), math.cos(10.0), math.sin(1000.0), math.sin(10.0)], [1, 1, 0, 0]])
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=0)


def test_the_timestep_embedding_is_float32_for_a_bf16_sigma() -> None:
    out = timestep_embedding(torch.tensor([0.5], dtype=torch.bfloat16), dim=8)

    assert out.shape == (1, 8) and out.dtype == torch.float32


def test_rope_at_position_zero_is_the_identity_rotation() -> None:
    out = rope(torch.zeros(1, 3), dim=4, theta=1000.0)

    assert out.shape == (1, 3, 2, 2, 2)
    torch.testing.assert_close(out, torch.eye(2).expand(1, 3, 2, 2, 2))


def test_rope_rotates_the_first_pair_by_the_position_and_slower_pairs_by_theta_powers() -> None:
    out = rope(torch.tensor([[1.0]]), dim=4, theta=100.0)

    angles = torch.atan2(out[0, 0, :, 1, 0], out[0, 0, :, 0, 0])
    torch.testing.assert_close(angles, torch.tensor([1.0, 0.1]))
    expected = torch.tensor([[math.cos(1.0), -math.sin(1.0)], [math.sin(1.0), math.cos(1.0)]])
    torch.testing.assert_close(out[0, 0, 0], expected)


def test_embed_nd_stacks_one_block_of_pairs_per_axis_with_a_broadcast_head_dimension() -> None:
    ids = torch.zeros(2, 5, 3)
    ids[..., 1] = 2.0

    pe = embed_nd(ids, [4, 6, 6], theta=1000.0)

    assert pe.shape == (2, 1, 5, 8, 2, 2)
    torch.testing.assert_close(pe[:, :, :, :2], torch.eye(2).expand(2, 1, 5, 2, 2, 2))
    torch.testing.assert_close(pe[0, 0, 0, 2, 1, 0], torch.tensor(math.sin(2.0)))


def test_apply_rope_turns_each_pair_by_its_matrix_and_keeps_the_dtype() -> None:
    pe = rope(torch.tensor([[math.pi / 2]]), dim=2, theta=1000.0)[:, None]
    x = torch.tensor([[[[3.0, 5.0]]]], dtype=torch.bfloat16)

    out = apply_rope1(x, pe)

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), torch.tensor([[[[-5.0, 3.0]]]]), atol=0.05, rtol=0)


def test_the_rms_norm_stores_its_scale_zero_centred_so_a_fresh_one_is_a_plain_rms_norm() -> None:
    norm = RMSNorm(2)

    out = norm(torch.tensor([[3.0, 4.0]]))

    rms = math.sqrt(12.5)
    torch.testing.assert_close(out, torch.tensor([[3.0 / rms, 4.0 / rms]]), atol=1e-4, rtol=0)
    norm.scale.data = torch.tensor([1.0, -0.5])
    torch.testing.assert_close(
        norm(torch.tensor([[3.0, 4.0]])), torch.tensor([[6.0 / rms, 2.0 / rms]]), atol=1e-4, rtol=0
    )


def test_the_rms_norm_returns_the_input_dtype() -> None:
    assert RMSNorm(4)(torch.ones(1, 4, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_the_swiglu_width_is_two_thirds_of_the_features_times_the_multiplier_rounded_up() -> None:
    plain = SwiGLU(30, 1, multiple=8)
    wide = SwiGLU(30, 3, multiple=8)

    assert plain.gate.out_features == plain.up.out_features == 24
    assert plain.down.in_features == 24 and plain.down.out_features == 30
    assert wide.gate.out_features == 64
    assert SwiGLU(96, 4).gate.out_features == 256 and SwiGLU(100, 1).gate.out_features == 128


def test_the_swiglu_gates_the_up_projection_with_silu_of_the_gate() -> None:
    mlp = SwiGLU(3, 1, multiple=1)
    for layer in (mlp.gate, mlp.up, mlp.down):
        layer.weight.data = torch.eye(layer.out_features, layer.in_features)
    x = torch.tensor([[1.0, -2.0, 5.0]])

    out = mlp(x)

    expected = torch.nn.functional.silu(x[:, :2]) * x[:, :2]
    torch.testing.assert_close(out, torch.cat([expected, torch.zeros(1, 1)], dim=1))
