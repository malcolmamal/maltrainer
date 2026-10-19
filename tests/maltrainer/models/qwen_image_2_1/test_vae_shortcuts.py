from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.qwen_image_2_1.vae import AvgDown3D, DecoderStage, DupUp3D, EncoderStage

SQUARE = torch.tensor([[1.0, 2.0], [3.0, 4.0]])


class _Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.factor


def test_the_average_shortcut_is_parameter_free_and_derives_its_group_size() -> None:
    shortcut = AvgDown3D(in_channels=4, out_channels=8, factor_t=2, factor_s=2)

    assert shortcut.factor == 8
    assert shortcut.group_size == 4
    assert list(shortcut.state_dict()) == []


def test_without_any_factor_the_average_shortcut_is_the_identity() -> None:
    x = torch.randn(2, 3, 4, 4, generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(AvgDown3D(3, 3, factor_t=1, factor_s=1)(x), x)


def test_a_spatial_factor_averages_each_channel_over_its_2x2_blocks() -> None:
    x = torch.stack([SQUARE, SQUARE * 10]).view(1, 2, 2, 2)

    out = AvgDown3D(2, 2, factor_t=1, factor_s=2)(x)

    torch.testing.assert_close(out, torch.tensor([2.5, 25.0]).view(1, 2, 1, 1))


def test_fewer_output_channels_average_consecutive_rows_of_each_block() -> None:
    out = AvgDown3D(1, 2, factor_t=1, factor_s=2)(SQUARE.view(1, 1, 2, 2))

    torch.testing.assert_close(out, torch.tensor([1.5, 3.5]).view(1, 2, 1, 1))


def test_a_temporal_factor_averages_the_single_frame_with_one_zero_frame() -> None:
    x = torch.randn(1, 1, 2, 2, generator=torch.Generator().manual_seed(0))

    out = AvgDown3D(1, 1, factor_t=2, factor_s=1)(x)

    torch.testing.assert_close(out, x / 2)


def test_the_average_shortcut_rejects_widths_that_do_not_divide() -> None:
    with pytest.raises(ValueError, match=r"in_channels \(1\) x factor \(1\)"):
        AvgDown3D(1, 3, factor_t=1, factor_s=1)


def test_the_duplicating_shortcut_is_parameter_free_and_derives_its_repeat_count() -> None:
    shortcut = DupUp3D(in_channels=4, out_channels=8, factor_t=2, factor_s=2)

    assert shortcut.repeats == 16
    assert list(shortcut.state_dict()) == []


def test_a_spatial_factor_unfolds_four_channels_into_a_2x2_block() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 4, 1, 1)

    out = DupUp3D(4, 1, factor_t=1, factor_s=2)(x)

    torch.testing.assert_close(out, SQUARE.view(1, 1, 2, 2))


def test_a_temporal_factor_keeps_only_the_last_duplicated_frame_of_a_single_image() -> None:
    x = torch.tensor([5.0, 9.0]).view(1, 2, 1, 1)

    out = DupUp3D(2, 1, factor_t=2, factor_s=1)(x)

    torch.testing.assert_close(out, torch.tensor([9.0]).view(1, 1, 1, 1))


def test_fewer_input_channels_are_repeated_to_fill_the_output_width() -> None:
    x = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1)

    out = DupUp3D(2, 4, factor_t=1, factor_s=1)(x)

    torch.testing.assert_close(out, torch.tensor([1.0, 1.0, 2.0, 2.0]).view(1, 4, 1, 1))


def test_the_duplicating_shortcut_rejects_widths_that_do_not_divide() -> None:
    with pytest.raises(ValueError, match=r"out_channels \(1\) x factor \(1\)"):
        DupUp3D(3, 1, factor_t=1, factor_s=1)


def test_duplicating_after_averaging_returns_a_block_constant_image() -> None:
    x = torch.arange(16.0).view(1, 1, 4, 4)

    out = DupUp3D(4, 1, factor_t=1, factor_s=2)(AvgDown3D(1, 4, factor_t=1, factor_s=2)(x))

    assert out.shape == x.shape
    torch.testing.assert_close(out[0, 0, :2, :2], torch.tensor([[0.0, 1.0], [4.0, 5.0]]))


@pytest.mark.parametrize(("stage_type", "attribute"), [(EncoderStage, "downsamples"), (DecoderStage, "upsamples")])
def test_a_stage_runs_its_blocks_then_the_resampler_and_adds_the_shortcut_of_its_input(
    stage_type: type, attribute: str
) -> None:
    stage = stage_type([_Scale(2.0), _Scale(3.0)], _Scale(5.0), _Scale(0.5))

    out = stage(torch.ones(1, 1, 1, 1))

    assert len(getattr(stage, attribute)) == 3
    torch.testing.assert_close(out, torch.full((1, 1, 1, 1), 30.5))


@pytest.mark.parametrize("stage_type", [EncoderStage, DecoderStage])
def test_a_stage_without_resampler_or_shortcut_only_runs_its_blocks(stage_type: type) -> None:
    stage = stage_type([_Scale(2.0), _Scale(3.0)])

    out = stage(torch.ones(1, 1, 1, 1))

    torch.testing.assert_close(out, torch.full((1, 1, 1, 1), 6.0))
    assert stage.avg_shortcut is None
