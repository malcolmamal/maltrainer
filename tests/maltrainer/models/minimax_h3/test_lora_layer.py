from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.minimax_h3.lora import GridAdaLNLoRA, MiniMaxH3LoRALayer


class _DtypeRecorder(nn.Linear):
    def __init__(self, features: int, rank: int) -> None:
        super().__init__(features, rank, bias=False)
        self.seen: list[torch.dtype] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen.append(x.dtype)
        return super().forward(x.to(self.weight.dtype))


def _hand_layer(alpha: float = 4.0) -> MiniMaxH3LoRALayer:
    layer = MiniMaxH3LoRALayer(nn.Linear(2, 1), rank=2, alpha=alpha)
    with torch.no_grad():
        layer.lora_down.weight.copy_(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))
        layer.lora_up.weight.copy_(torch.tensor([[2.0, 3.0]]))
    return layer


def test_a_layer_is_a_bias_free_down_up_pair_and_scales_by_alpha_over_rank() -> None:
    layer = MiniMaxH3LoRALayer(nn.Linear(3, 5), rank=2, alpha=6.0)

    assert layer.lora_down.weight.shape == (2, 3)
    assert layer.lora_up.weight.shape == (5, 2)
    assert layer.lora_down.bias is None
    assert layer.lora_up.bias is None
    assert (layer.rank, layer.alpha, layer.scale) == (2, 6.0, 3.0)


def test_a_fresh_layer_adds_exactly_nothing() -> None:
    torch.manual_seed(0)
    layer = MiniMaxH3LoRALayer(nn.Linear(3, 5), rank=2, alpha=1.0)

    out = layer(torch.randn(4, 3))

    assert layer.lora_down.weight.abs().sum() > 0
    assert torch.equal(out, torch.zeros(4, 5))


def test_the_output_is_up_of_down_times_alpha_over_rank() -> None:
    layer = _hand_layer(alpha=4.0)

    out = layer(torch.tensor([[1.0, 2.0]]))

    assert out.item() == (2.0 * 1.0 + 3.0 * 3.0) * (4.0 / 2.0)


def test_the_activation_is_cast_up_to_the_adapter_dtype() -> None:
    layer = MiniMaxH3LoRALayer(nn.Linear(2, 2), rank=1, alpha=1.0).to(torch.float64)

    out = layer(torch.ones(1, 2, dtype=torch.bfloat16))

    assert out.dtype == torch.float64


def test_an_integer_activation_is_left_for_the_linear_to_reject() -> None:
    layer = MiniMaxH3LoRALayer(nn.Linear(2, 2), rank=1, alpha=1.0)

    with pytest.raises(RuntimeError, match="dtype"):
        layer(torch.ones(1, 2, dtype=torch.long))


@pytest.mark.parametrize(("autocast", "expected"), [(False, torch.float32), (True, torch.bfloat16)])
def test_the_activation_is_not_upcast_under_autocast(
    monkeypatch: pytest.MonkeyPatch, autocast: bool, expected: torch.dtype
) -> None:
    layer = MiniMaxH3LoRALayer(nn.Linear(2, 2), rank=1, alpha=1.0)
    layer.lora_down = _DtypeRecorder(2, 1)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda *_args: autocast)

    layer(torch.ones(1, 2, dtype=torch.bfloat16))

    assert layer.lora_down.seen == [expected]


def test_a_grid_row_is_chosen_by_the_nearest_table_row_and_scaled_through_down_and_up() -> None:
    table = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    grid = torch.arange(12.0).view(3, 4)
    down, up = torch.randn(2, 4, generator=torch.Generator().manual_seed(0)), torch.randn(5, 2)
    layer = GridAdaLNLoRA(down, up, scale=0.5, table=table, grid=grid)

    out = layer(torch.tensor([[0.9, 0.1], [0.1, 0.8], [0.0, 0.2]]))

    expected = grid[[1, 2, 0]] @ down.T @ up.T * 0.5
    torch.testing.assert_close(out, expected)


def test_a_grid_layer_keeps_detached_float32_copies_of_everything_it_was_given() -> None:
    down = torch.ones(2, 4, dtype=torch.float16, requires_grad=True)
    table = torch.zeros(2, 2)
    layer = GridAdaLNLoRA(down, torch.ones(3, 2), 1.0, table, torch.zeros(2, 4))

    table += 1.0

    assert all(buffer.dtype == torch.float32 for buffer in layer.buffers())
    assert not layer.down.requires_grad
    assert layer.table.sum() == 0.0
    assert set(layer.state_dict()) == {"down", "up", "table", "grid"}
