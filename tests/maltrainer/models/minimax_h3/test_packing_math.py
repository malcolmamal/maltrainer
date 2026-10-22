"""Defaults and the temporal patch of the H3 packing helpers, pinned (mutation testing)."""

from __future__ import annotations

import pytest
import torch

from maltrainer.models.minimax_h3.packing import apply_flow_shift, patchify_video


def test_the_default_flow_shift_is_12() -> None:
    t = torch.tensor([0.25, 0.5])

    torch.testing.assert_close(apply_flow_shift(t), torch.tensor([3.0 / 3.75, 6.0 / 6.5]))


@pytest.mark.parametrize("shift", [1.0, 2.0])
def test_a_flow_shift_is_the_closed_form(shift: float) -> None:
    t = torch.tensor([0.5])

    torch.testing.assert_close(apply_flow_shift(t, shift), torch.tensor([shift * 0.5 / (1.0 + (shift - 1.0) * 0.5)]))


def test_a_temporal_patch_groups_frames_before_rows_and_columns() -> None:
    latent = torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(1, 1, 4, 2, 2)  # [B, C, T, H, W], one value per cell

    tokens = patchify_video(latent, patch_size=(2, 2, 2))

    assert tokens.shape == (1, 2, 8)  # two temporal groups of one 2x2 patch each, 2 frames x 4 cells per token
    torch.testing.assert_close(tokens[0, 0], torch.arange(8, dtype=torch.float32))
    torch.testing.assert_close(tokens[0, 1], torch.arange(8, 16, dtype=torch.float32))
