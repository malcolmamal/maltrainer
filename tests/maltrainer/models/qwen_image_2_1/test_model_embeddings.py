from __future__ import annotations

import math

import pytest
import torch

from maltrainer.models.qwen_image_2_1.model import (
    RMSNorm,
    SwiGLUFeedForward,
    TemporalTimesteps,
    TextProjection,
    TimestepEmbedder,
    TimestepProjEmbeddings,
    ZeroCenterRMSNorm,
)


def test_the_sinusoidal_embedding_is_cosines_then_sines_at_geometric_frequencies() -> None:
    proj = TemporalTimesteps(timestep_dim=4, max_period=10000, time_factor=1.0)

    out = proj(torch.tensor([2.0]))

    torch.testing.assert_close(
        out, torch.tensor([[math.cos(2.0), math.cos(0.02), math.sin(2.0), math.sin(0.02)]]), atol=1e-6, rtol=0
    )


def test_the_timestep_is_scaled_by_the_time_factor_before_embedding() -> None:
    scaled = TemporalTimesteps(timestep_dim=4, time_factor=1000.0)(torch.tensor([0.002]))
    plain = TemporalTimesteps(timestep_dim=4, time_factor=1.0)(torch.tensor([2.0]))

    torch.testing.assert_close(scaled, plain)


def test_the_embedding_is_float32_whatever_the_timestep_dtype_and_has_no_saved_state() -> None:
    proj = TemporalTimesteps(timestep_dim=8)

    out = proj(torch.tensor([0.5, 1.0], dtype=torch.bfloat16))

    assert out.shape == (2, 8)
    assert out.dtype == torch.float32
    assert proj.state_dict() == {}


def test_the_timestep_embedder_is_a_bias_free_two_layer_silu_mlp() -> None:
    embedder = TimestepEmbedder(in_channels=3, time_embed_dim=5)
    x = torch.randn(2, 3)

    out = embedder(x)

    assert [name for name, _ in embedder.named_parameters()] == ["linear_1.weight", "linear_2.weight"]
    torch.testing.assert_close(
        out, torch.nn.functional.silu(x @ embedder.linear_1.weight.T) @ embedder.linear_2.weight.T
    )


def test_the_timestep_projection_casts_the_embedding_to_the_hidden_state_dtype() -> None:
    block = TimestepProjEmbeddings(embedding_dim=8).to(torch.bfloat16)

    out = block(torch.tensor([0.1, 0.9]), torch.zeros(1, 4, dtype=torch.bfloat16))

    assert out.shape == (2, 8)
    assert out.dtype == torch.bfloat16


def test_a_zero_centred_norm_with_a_zero_weight_divides_by_the_root_mean_square() -> None:
    norm = ZeroCenterRMSNorm(2, eps=0.0)

    out = norm(torch.tensor([[3.0, 4.0]]))

    torch.testing.assert_close(out, torch.tensor([[3.0, 4.0]]) / math.sqrt(12.5))


def test_the_zero_centred_weight_is_an_offset_from_one_and_the_dtype_is_kept() -> None:
    norm = ZeroCenterRMSNorm(2, eps=0.0)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, -1.0]))

    out = norm(torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16))

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), torch.tensor([[6.0, 0.0]]) / math.sqrt(12.5), atol=0.02, rtol=0.02)


def test_the_qk_norm_rescales_each_vector_to_length_root_dim_times_its_weight() -> None:
    norm = RMSNorm(2, eps=1e-12)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.0, 2.0]))

    out = norm(torch.tensor([[3.0, 4.0], [0.0, 5.0]]))

    torch.testing.assert_close(out, torch.tensor([[0.6, 1.6], [0.0, 2.0]]) * math.sqrt(2.0))


def test_the_swiglu_feed_forward_gates_the_up_half_with_silu_of_the_gate_half() -> None:
    mlp = SwiGLUFeedForward(hidden_size=2, mlp_hidden_size=2)
    with torch.no_grad():
        mlp.gate_up.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]))
        mlp.out.weight.copy_(torch.eye(2))

    out = mlp(torch.tensor([[2.0, 1.0]]))

    gate = torch.nn.functional.silu(torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(out, (gate * torch.tensor([3.0, 1.0]))[None])


@pytest.mark.parametrize("batch", [1, 3])
def test_the_text_projection_maps_the_context_width_to_the_hidden_width(batch: int) -> None:
    torch.manual_seed(0)
    projection = TextProjection(context_in_dim=6, hidden_size=4)

    out = projection(torch.randn(batch, 5, 6))

    assert out.shape == (batch, 5, 4)
    assert sorted(name for name, _ in projection.named_parameters()) == [
        "in_layer.weight",
        "out_layer.weight",
        "text_norm.weight",
    ]
