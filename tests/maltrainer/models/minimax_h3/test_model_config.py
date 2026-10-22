from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from maltrainer.models.minimax_h3.model import (
    MINIMAX_H3_MODALITY_COUNT,
    MiniMaxH3AdaLNProjection,
    MiniMaxH3TimeEmbedder,
    MiniMaxH3TokenTag,
    MiniMaxH3TransformerConfig,
    get_timestep_embedding,
)


class _Probe(nn.Module):
    def __init__(self, scale_weight: torch.Tensor | None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        if scale_weight is not None:
            self.scale_weight = scale_weight
        self.seen: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x)
        return x


def test_modality_tags_name_the_three_token_kinds_the_adaln_table_is_indexed_by() -> None:
    assert (MiniMaxH3TokenTag.VIDEO, MiniMaxH3TokenTag.TEXT, MiniMaxH3TokenTag.AUDIO) == (0, 1, 2)
    assert len(MiniMaxH3TokenTag) == MINIMAX_H3_MODALITY_COUNT


def test_the_default_config_describes_the_full_h3_transformer() -> None:
    config = MiniMaxH3TransformerConfig()

    assert config.num_attention_heads * config.attention_head_dim == 7168
    assert (config.hidden_size, config.num_layers, config.patch_size) == (5376, 50, (1, 2, 2))
    assert config.adaln_t_table_size is None


def test_from_dict_reads_known_fields_turns_patch_size_into_a_tuple_and_ignores_the_rest() -> None:
    config = MiniMaxH3TransformerConfig.from_dict({"hidden_size": 8, "patch_size": [1, 4, 4], "unknown": "ignored"})

    assert config.hidden_size == 8
    assert config.patch_size == (1, 4, 4)
    assert config.num_layers == 50


def test_from_dict_without_a_patch_size_keeps_the_default() -> None:
    assert MiniMaxH3TransformerConfig.from_dict({}).patch_size == (1, 2, 2)


def test_a_timestep_embedding_is_sines_then_cosines_of_geometric_frequencies() -> None:
    out = get_timestep_embedding(torch.tensor([0.0, 1.0]), 4)

    expected = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [math.sin(1.0), math.sin(0.01), math.cos(1.0), math.cos(0.01)]],
    )
    torch.testing.assert_close(out, expected)


def test_flipping_sin_to_cos_swaps_the_two_halves() -> None:
    steps = torch.tensor([0.3, 2.0])

    plain = get_timestep_embedding(steps, 6)
    flipped = get_timestep_embedding(steps, 6, flip_sin_to_cos=True)

    torch.testing.assert_close(flipped, torch.cat([plain[:, 3:], plain[:, :3]], dim=-1))


def test_an_odd_embedding_width_is_zero_padded_on_the_right() -> None:
    out = get_timestep_embedding(torch.tensor([0.5]), 5)

    assert out.shape == (1, 5)
    assert out[0, 4].item() == 0.0


def test_scale_and_max_period_change_the_frequencies() -> None:
    out = get_timestep_embedding(torch.tensor([1.0]), 4, scale=2.0, max_period=100)

    expected = [math.sin(2.0), math.sin(2.0 * 0.1), math.cos(2.0), math.cos(2.0 * 0.1)]
    torch.testing.assert_close(out, torch.tensor([expected]))


def test_a_timestep_embedding_needs_a_one_dimensional_tensor() -> None:
    with pytest.raises(ValueError, match="1d-tensor"):
        get_timestep_embedding(torch.zeros(2, 1), 4)


def test_the_time_embedder_runs_the_flipped_sinusoid_through_two_linears_with_a_silu() -> None:
    torch.manual_seed(0)
    embedder = MiniMaxH3TimeEmbedder(freq_dim=8, hidden_dim=5, output_dim=7)
    steps = torch.tensor([0.1, 0.9, 1.0])

    out = embedder(steps)

    sinusoid = get_timestep_embedding(steps, 8, flip_sin_to_cos=True)
    expected = embedder.proj_out(torch.nn.functional.silu(embedder.proj_in(sinusoid)))
    assert out.shape == (3, 7)
    torch.testing.assert_close(out, expected)


def test_the_time_embedder_casts_the_sinusoid_to_the_weight_dtype() -> None:
    embedder = MiniMaxH3TimeEmbedder(freq_dim=8, hidden_dim=5, output_dim=7).to(torch.bfloat16)

    assert embedder(torch.tensor([0.5])).dtype == torch.bfloat16


@pytest.mark.parametrize("apply_silu", [True, False])
def test_the_adaln_projection_applies_silu_only_when_asked(apply_silu: bool) -> None:
    torch.manual_seed(0)
    projection = MiniMaxH3AdaLNProjection(4, 6, apply_silu=apply_silu)
    x = torch.randn(3, 4)

    out = projection(x)

    activated = torch.nn.functional.silu(x) if apply_silu else x
    torch.testing.assert_close(out, projection.linear(activated))


def test_the_adaln_projection_feeds_the_linear_in_the_dtype_of_its_weight() -> None:
    projection = MiniMaxH3AdaLNProjection(2, 2)
    projection.linear = _Probe(None)

    projection(torch.ones(1, 2))

    assert projection.linear.seen[0].dtype == torch.bfloat16


def test_a_quantized_linear_is_fed_in_the_dtype_of_its_scale_weight() -> None:
    projection = MiniMaxH3AdaLNProjection(2, 2)
    projection.linear = _Probe(torch.zeros(1, dtype=torch.float16))

    projection(torch.ones(1, 2))

    assert projection.linear.seen[0].dtype == torch.float16
