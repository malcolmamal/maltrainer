from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from maltrainer.models.minimax_h3.audio_vae import (
    AttnProjection,
    CausalAttention,
    DacEncoder,
    EncoderBlock,
    GeGluMlp,
    ResidualUnit,
    Snake1d,
    _fold_weight_norm,
)


def _random(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    for p in module.parameters():
        p.data = torch.randn_like(p) * 0.3
    return module.eval()


def test_snake_adds_the_squared_sine_scaled_by_the_inverse_alpha() -> None:
    snake = Snake1d(2)
    snake.alpha.data = torch.tensor([[[2.0], [0.5]]])
    x = torch.tensor([[[0.0, 1.0, -1.5], [0.0, 1.0, -1.5]]])

    out = snake(x)

    for channel, alpha in enumerate((2.0, 0.5)):
        expected = [v + math.sin(alpha * v) ** 2 / alpha for v in (0.0, 1.0, -1.5)]
        assert out[0, channel].tolist() == pytest.approx(expected, abs=1e-6)


def test_snake_runs_in_the_input_dtype_whatever_dtype_its_alpha_is() -> None:
    out = Snake1d(3)(torch.randn(1, 3, 5, dtype=torch.bfloat16))

    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("dilation", [1, 3, 9])
def test_a_residual_unit_keeps_the_length_whatever_the_dilation_and_adds_its_input(dilation: int) -> None:
    unit = _random(ResidualUnit(4, dilation))
    x = torch.randn(2, 4, 40)

    out = unit(x)

    assert out.shape == x.shape
    torch.testing.assert_close(out, unit.block(x) + x)


@pytest.mark.parametrize("stride", [2, 4, 5])
def test_an_encoder_block_halves_the_channels_in_and_divides_the_length_by_the_stride(stride: int) -> None:
    block = _random(EncoderBlock(8, stride))
    x = torch.randn(1, 4, 20 * stride)

    out = block(x)

    assert out.shape == (1, 8, 20)


def test_the_dac_encoder_downsamples_by_the_product_of_its_strides_to_the_latent_width() -> None:
    encoder = _random(DacEncoder(d_model=2, strides=(2, 4, 4, 5, 5), d_latent=12))

    out = encoder(torch.randn(1, 1, 1600))

    assert out.shape == (1, 12, 2)  # 800 samples per frame


def test_the_geglu_mlp_gates_a_tanh_gelu_branch_with_a_linear_one_after_layer_norm() -> None:
    mlp = _random(GeGluMlp(6, 10))
    x = torch.randn(2, 3, 6)

    normed = F.layer_norm(x, (6,), mlp.norm.weight, mlp.norm.bias)
    expected = mlp.w2(F.gelu(mlp.w0(normed), approximate="tanh") * mlp.w1(normed))

    torch.testing.assert_close(mlp(x), expected)


def test_causal_attention_pools_heads_to_the_output_width_and_ignores_the_future() -> None:
    attn = _random(CausalAttention(in_dim=16, out_dim=8, num_heads=2))
    x = torch.randn(1, 6, 16)
    changed = x.clone()
    changed[:, 4:] += 3.0

    out, out_changed = attn(x), attn(changed)

    assert out.shape == (1, 6, 8)
    torch.testing.assert_close(out[:, :4], out_changed[:, :4])
    assert not torch.allclose(out[:, 4:], out_changed[:, 4:])


def test_causal_attention_applies_q_and_v_biases_but_never_a_key_bias() -> None:
    attn = _random(CausalAttention(in_dim=16, out_dim=8, num_heads=2))
    x = torch.randn(1, 5, 16)
    baseline = attn(x)

    attn.zero_k_bias.fill_(10.0)  # softmax is shift-invariant in k's bias only; q/v biases are not
    key_shifted = attn(x)
    attn.v_bias.data += 1.0
    value_shifted = attn(x)

    torch.testing.assert_close(key_shifted, baseline, atol=1e-5, rtol=1e-4)
    assert not torch.allclose(value_shifted, baseline)


def test_attention_projection_adds_a_linear_skip_a_pooled_attention_and_a_mlp_residual() -> None:
    block = _random(AttnProjection(in_dim=16, out_dim=8, num_heads=2))
    x = torch.randn(2, 5, 16)

    mid = block.proj(block.norm3(x)) + block.attn(block.norm1(x))
    expected = mid + block.mlp(block.norm2(mid))

    out = block(x)
    assert out.shape == (2, 5, 8)
    torch.testing.assert_close(out, expected)


def test_weight_norm_pairs_fold_into_a_weight_with_the_stored_magnitude_per_output_row() -> None:
    v = torch.randn(3, 2, 4)
    g = torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1)
    state = {"a.weight_g": g, "a.weight_v": v, "a.bias": torch.ones(3), "b.weight": torch.zeros(1)}

    folded = _fold_weight_norm(state)

    assert set(folded) == {"a.weight", "a.bias", "b.weight"}
    torch.testing.assert_close(folded["a.weight"].flatten(1).norm(dim=1), torch.tensor([1.0, 2.0, 3.0]))
    direction = v.flatten(1) / v.flatten(1).norm(dim=1, keepdim=True)
    torch.testing.assert_close(folded["a.weight"].flatten(1) / g.flatten()[:, None], direction)
    assert folded["a.bias"] is state["a.bias"]


class _Shorten(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x[..., 3:-3])


def test_a_residual_unit_whose_block_shortens_the_signal_crops_the_skip_to_match() -> None:
    unit = ResidualUnit(2, dilation=1)
    unit.block = _Shorten()
    x = torch.arange(20.0).view(1, 1, 20).expand(1, 2, 20)

    out = unit(x)

    assert out.shape == (1, 2, 14)
    torch.testing.assert_close(out, 1.0 + x[..., 3:-3])
