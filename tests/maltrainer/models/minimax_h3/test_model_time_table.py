from __future__ import annotations

import pytest
import torch

from maltrainer.models.minimax_h3.model import MiniMaxH3Transformer
from tests.maltrainer.models.minimax_h3.h3_tiny import Packed, tiny_config, tiny_transformer


def _with_table(rows: torch.Tensor) -> MiniMaxH3Transformer:
    model = tiny_transformer(adaln_t_table_size=rows.shape[0])
    model.adaln_t_table.copy_(rows)
    return model


def test_a_table_model_replaces_the_time_embedder_with_a_persistent_curve_table() -> None:
    model = tiny_transformer(adaln_t_table_size=4)

    assert model.time_embedder is None
    assert model.state_dict()["adaln_t_table"].shape == (4, 10)
    assert not any(key.startswith("time_embedder") for key in model.state_dict())


def test_only_the_full_model_applies_a_silu_before_the_adaln_projections() -> None:
    full, pruned = tiny_transformer(), tiny_transformer(adaln_t_table_size=4)

    assert (full.blocks[0].adaln_proj.apply_silu, full.final_layer.adaln_proj.apply_silu) == (True, True)
    assert (pruned.blocks[0].adaln_proj.apply_silu, pruned.final_layer.adaln_proj.apply_silu) == (False, False)


@pytest.mark.parametrize("size", [0, 1])
def test_a_curve_table_needs_at_least_two_rows(size: int) -> None:
    with pytest.raises(ValueError, match="at least two rows"):
        MiniMaxH3Transformer(tiny_config(adaln_t_table_size=size))


def test_timesteps_between_grid_points_blend_the_two_neighbouring_rows() -> None:
    rows = torch.arange(3.0)[:, None] * torch.ones(3, 10)
    model = _with_table(rows)

    out = model._time_embedding(torch.tensor([0.25, 0.75, 0.5]))

    torch.testing.assert_close(out[:, 0], torch.tensor([0.5, 1.5, 1.0]))
    assert out.shape == (3, 10)


def test_timesteps_on_a_grid_point_return_exactly_that_row() -> None:
    rows = torch.randn(4, 10, generator=torch.Generator().manual_seed(0))
    model = _with_table(rows)

    out = model._time_embedding(torch.tensor([0.0, 1 / 3, 1.0]))

    torch.testing.assert_close(out, rows[[0, 1, 3]])


def test_timesteps_outside_zero_to_one_are_clamped_to_the_end_rows() -> None:
    rows = torch.randn(3, 10, generator=torch.Generator().manual_seed(1))
    model = _with_table(rows)

    out = model._time_embedding(torch.tensor([-0.5, 7.0]))

    torch.testing.assert_close(out, rows[[0, 2]])


def test_a_table_model_runs_and_follows_its_table() -> None:
    rows = torch.randn(3, 10, generator=torch.Generator().manual_seed(2))
    model = _with_table(rows)
    inputs = Packed(text=2, audio=1, video=2).inputs()

    first = model(**inputs)
    model.adaln_t_table.mul_(0.0)
    second = model(**inputs)

    assert (first.video.shape, first.audio.shape) == ((1, 2, 8), (1, 1, 3))
    assert not torch.allclose(first.video, second.video, atol=1e-4)
