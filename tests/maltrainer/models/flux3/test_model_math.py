from __future__ import annotations

import math
import re

import pytest
import torch

from maltrainer.models.flux3.model import (
    Flux3Config,
    Flux3Transformer,
    apply_rope,
    image_ids,
    rope,
    rope_from_ids,
    text_ids,
    timestep_embedding,
)
from tests.maltrainer.models.flux3.tiny_model import TINY, tiny_transformer

CPU = torch.device("cpu")


def test_the_timestep_embedding_is_cos_then_sin_of_a_thousand_times_sigma() -> None:
    out = timestep_embedding(torch.tensor([1.0, 0.0]), dim=4)

    expected = torch.tensor([[math.cos(1000.0), math.cos(10.0), math.sin(1000.0), math.sin(10.0)], [1, 1, 0, 0]])
    torch.testing.assert_close(out, expected, atol=2e-5, rtol=0)


def test_the_timestep_embedding_keeps_the_dtype_of_the_timesteps() -> None:
    out = timestep_embedding(torch.tensor([0.5, 1.0], dtype=torch.bfloat16), dim=256)

    assert out.shape == (2, 256)
    assert out.dtype == torch.bfloat16


def test_rope_at_position_zero_is_the_identity_rotation() -> None:
    out = rope(torch.zeros(3), dim=4, theta=10000)

    assert out.shape == (3, 2, 2, 2)
    torch.testing.assert_close(out, torch.eye(2).expand(3, 2, 2, 2))


def test_rope_rotates_the_first_frequency_pair_by_the_position_in_radians() -> None:
    out = rope(torch.tensor([2.0]), dim=2, theta=10000)

    expected = torch.tensor([[[[math.cos(2.0), -math.sin(2.0)], [math.sin(2.0), math.cos(2.0)]]]])
    torch.testing.assert_close(out, expected)


def test_rope_frequencies_fall_off_as_theta_to_the_minus_two_k_over_dim() -> None:
    out = rope(torch.tensor([1.0]), dim=4, theta=100)

    angles = torch.atan2(out[0, :, 1, 0], out[0, :, 0, 0])
    torch.testing.assert_close(angles, torch.tensor([1.0, 0.1]))


def test_rope_from_ids_gives_every_axis_its_own_32_dims_of_the_64_pairs() -> None:
    ids = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])

    pe = rope_from_ids(ids)

    assert pe.shape == (1, 1, 2, 64, 2, 2)
    torch.testing.assert_close(pe[0, 0, :, :48], torch.eye(2).expand(2, 48, 2, 2))
    assert torch.equal(pe[0, 0, 0, 48:], torch.eye(2).expand(16, 2, 2))
    torch.testing.assert_close(pe[0, 0, 1, 48, 0, 0], torch.tensor(math.cos(2.0)))
    torch.testing.assert_close(pe[0, 0, 1, 48, 1, 0], torch.tensor(math.sin(2.0)))


def test_apply_rope_turns_each_channel_pair_by_the_rotation_matrix() -> None:
    pe = rope(torch.tensor([math.pi / 2]), dim=2, theta=10000)[None, None]
    x = torch.tensor([[[[3.0, 5.0]]]])

    out = apply_rope(x, pe)

    torch.testing.assert_close(out, torch.tensor([[[[-5.0, 3.0]]]]), atol=1e-6, rtol=0)


def test_apply_rope_returns_the_dtype_of_its_input() -> None:
    pe = rope(torch.arange(2.0), dim=2, theta=10000)[None, None]

    out = apply_rope(torch.ones(1, 1, 2, 2, dtype=torch.bfloat16), pe)

    assert out.dtype == torch.bfloat16


def test_image_ids_scale_rows_and_columns_by_a_half_and_carry_the_reference_index() -> None:
    ids = image_ids(2, 3, index=5, device=CPU)

    assert ids.shape == (6, 4)
    assert ids.tolist() == [[0, r * 0.5, c * 0.5, 5] for r in range(2) for c in range(3)]


def test_text_ids_advance_along_the_last_axis_only() -> None:
    ids = text_ids(3, CPU)

    assert ids.tolist() == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]


def test_the_config_is_read_back_from_a_state_dict_of_any_variant() -> None:
    config = Flux3Config(hidden_size=128, num_heads=1, depth=2, depth_single_blocks=3, fused_qkv=True, flow_map=True)

    found = Flux3Config.from_state_dict(Flux3Transformer(config).state_dict())

    assert found == config


def test_the_config_of_a_plain_checkpoint_has_split_projections_and_no_flow_map() -> None:
    found = Flux3Config.from_state_dict(tiny_transformer().state_dict())

    assert found == TINY
    assert (found.fused_qkv, found.flow_map) == (False, False)


def test_a_state_dict_without_the_flux3_blocks_is_not_a_flux3_checkpoint() -> None:
    with pytest.raises(ValueError, match=re.escape("Not a FLUX.3 DiT checkpoint")):
        Flux3Config.from_state_dict({"single_blocks.0.q_proj.weight": torch.zeros(1)})


def test_rope_from_ids_is_each_axis_rope_at_theta_10000_over_32_dims() -> None:
    ids = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [0.5, 0.0, 9.0, 1.5]])

    pe = rope_from_ids(ids)

    for axis in range(4):
        torch.testing.assert_close(pe[0, 0, :, 16 * axis : 16 * axis + 16], rope(ids[:, axis], 32, 10000))
