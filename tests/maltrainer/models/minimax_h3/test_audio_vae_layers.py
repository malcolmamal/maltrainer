from __future__ import annotations

import math

import torch

from maltrainer.models.minimax_h3.audio_vae import (
    HOP_LENGTH,
    LATENTS_PER_SECOND,
    SAMPLE_RATE,
    AttnProjection,
    CausalAttention,
    DacEncoder,
    EncoderBlock,
    GeGluMlp,
    ResidualUnit,
    Snake1d,
)
from tests.maltrainer.models.minimax_h3.h3_audio import STRIDES


def test_the_hop_is_the_product_of_the_encoder_strides_giving_forty_latents_per_second() -> None:
    assert math.prod(STRIDES) == HOP_LENGTH == 800
    assert SAMPLE_RATE // HOP_LENGTH == LATENTS_PER_SECOND == 40


def test_snake_adds_the_squared_sine_of_the_scaled_input_over_alpha() -> None:
    snake = Snake1d(1)
    with torch.no_grad():
        snake.alpha.fill_(2.0)
    x = torch.tensor([[[0.0, 0.5, 1.0]]])

    out = snake(x)

    torch.testing.assert_close(out, x + torch.sin(2.0 * x) ** 2 / 2.0)


def test_snake_follows_the_dtype_of_its_input() -> None:
    assert Snake1d(2)(torch.zeros(1, 2, 3, dtype=torch.float64)).dtype == torch.float64


def test_a_residual_unit_keeps_the_length_and_adds_its_input() -> None:
    unit = ResidualUnit(4, dilation=3)
    torch.nn.init.zeros_(unit.block[3].weight)
    torch.nn.init.zeros_(unit.block[3].bias)
    x = torch.randn(1, 4, 30)

    torch.testing.assert_close(unit(x), x)


def test_a_residual_unit_centre_crops_its_input_when_the_branch_is_shorter() -> None:
    unit = ResidualUnit(1, dilation=1)
    unit.block[1].padding = (0,)
    torch.nn.init.zeros_(unit.block[3].weight)
    torch.nn.init.zeros_(unit.block[3].bias)
    x = torch.arange(10.0).view(1, 1, 10)

    out = unit(x)

    assert out.flatten().tolist() == [3.0, 4.0, 5.0, 6.0]


def test_an_encoder_block_divides_the_length_by_its_stride_and_doubles_the_width() -> None:
    block = EncoderBlock(8, stride=5)

    assert block(torch.randn(1, 4, 100)).shape == (1, 8, 20)


def test_the_dac_encoder_turns_a_waveform_into_one_vector_per_hop() -> None:
    encoder = DacEncoder(2, STRIDES, 8)

    assert encoder(torch.randn(1, 1, 1600)).shape == (1, 8, 2)


def test_causal_attention_pools_the_width_and_ignores_later_positions() -> None:
    torch.manual_seed(0)
    attention = CausalAttention(8, 4, num_heads=2)
    x = torch.randn(1, 5, 8)
    later = x.clone()
    later[:, 3:] += 4.0

    out = attention(x)

    assert out.shape == (1, 5, 4)
    torch.testing.assert_close(out[:, :3], attention(later)[:, :3])
    assert not torch.allclose(out[:, 3:], attention(later)[:, 3:])


def test_a_ge_glu_mlp_gates_a_tanh_gelu_branch_with_a_linear_one() -> None:
    mlp = GeGluMlp(4, 6)
    x = torch.randn(2, 3, 4)

    normed = mlp.norm(x)
    expected = mlp.w2(torch.nn.functional.gelu(mlp.w0(normed), approximate="tanh") * mlp.w1(normed))

    torch.testing.assert_close(mlp(x), expected)


def test_the_attention_projection_maps_the_wide_features_to_the_latent_width() -> None:
    assert AttnProjection(8, 4, num_heads=2)(torch.randn(2, 5, 8)).shape == (2, 5, 4)
