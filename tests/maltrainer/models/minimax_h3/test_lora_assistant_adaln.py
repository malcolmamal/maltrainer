from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.minimax_h3.lora import GridAdaLNLoRA, MiniMaxH3LoRANetwork
from tests.maltrainer.models.minimax_h3.h3_adapters import hand_pair, write_adapter
from tests.maltrainer.models.minimax_h3.h3_tiny import tiny_transformer

ADALN = "blocks.0.adaln_proj.linear"
FULL_WIDTH = 12
LOGGER = "maltrainer.models.minimax_h3.lora"


def _pruned_network() -> MiniMaxH3LoRANetwork:
    transformer = tiny_transformer(adaln_t_table_size=3)
    transformer.adaln_t_table.copy_(torch.randn(3, 10, generator=torch.Generator().manual_seed(7)))
    return MiniMaxH3LoRANetwork(transformer, rank=2, alpha=4.0, block_range=(1, 1))


def _grid(path: Path, rows: int = 3) -> Path:
    grid = torch.randn(rows, FULL_WIDTH, generator=torch.Generator().manual_seed(8))
    save_file({"silu_t_emb_grid": grid}, str(path))
    return path


def _adapter(tmp_path: Path, modules: tuple[str, ...] = (ADALN,)) -> Path:
    pair = hand_pair(FULL_WIDTH, 6 * 16 * 3)
    return write_adapter(tmp_path / "assist.safetensors", dict.fromkeys(modules, (*pair, 2.0)))


def test_a_full_model_adaln_row_is_run_on_the_grid_row_nearest_to_the_table_input(tmp_path: Path) -> None:
    network = _pruned_network()
    path = _adapter(tmp_path)
    grid_path = _grid(tmp_path / "grid.safetensors")

    attached = network.load_assistant_adapter(str(path), multiplier=0.5, adaln_grid_path=str(grid_path))

    module = network.transformer.blocks[0].adaln_proj.linear
    table = network.transformer.adaln_t_table
    layer = network.assistant_layers["blocks_0_adaln_proj_linear"]
    x = table[[2, 0]] + 1e-3
    base = torch.nn.functional.linear(x, module.weight, module.bias)
    assert attached == 1
    assert isinstance(layer, GridAdaLNLoRA)
    torch.testing.assert_close(module(x), base + layer(x))
    assert layer.scale == (2.0 / 2) * 0.5
    torch.testing.assert_close(layer(x), (layer.grid[[2, 0]] @ layer.down.T @ layer.up.T) * layer.scale)


def test_a_grid_layer_stays_in_float32_whatever_dtype_is_asked_for(tmp_path: Path) -> None:
    network = _pruned_network()
    grid_path = _grid(tmp_path / "grid.safetensors")

    network.load_assistant_adapter(
        str(_adapter(tmp_path)), device=torch.device("cpu"), dtype=torch.bfloat16, adaln_grid_path=str(grid_path)
    )

    layer = network.assistant_layers["blocks_0_adaln_proj_linear"]
    assert {buffer.dtype for buffer in layer.buffers()} == {torch.float32}


def test_the_grid_is_read_once_for_all_the_adaln_rows(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    network = MiniMaxH3LoRANetwork(
        tiny_transformer(adaln_t_table_size=3), rank=2, alpha=4.0, block_range=(2, 2), include_token_refiner=False
    )
    grid_path = _grid(tmp_path / "grid.safetensors")
    path = _adapter(tmp_path, ("blocks.0.adaln_proj.linear", "blocks.1.adaln_proj.linear"))

    with caplog.at_level("INFO", logger=LOGGER):
        attached = network.load_assistant_adapter(str(path), adaln_grid_path=str(grid_path))

    assert attached == 2
    assert "2 modules (2 AdaLN via the t_emb grid)" in caplog.text


def test_adaln_rows_are_skipped_with_a_warning_when_no_grid_is_given(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    network = _pruned_network()
    x = network.transformer.adaln_t_table[:2].clone()
    before = network.transformer.blocks[0].adaln_proj.linear(x)

    with caplog.at_level("WARNING", logger=LOGGER):
        attached = network.load_assistant_adapter(str(_adapter(tmp_path)))

    assert attached == 0
    assert "1 full-model AdaLN row(s)" in caplog.text
    torch.testing.assert_close(network.transformer.blocks[0].adaln_proj.linear(x), before)


def test_a_grid_from_another_build_is_rejected(tmp_path: Path) -> None:
    network = _pruned_network()
    grid_path = _grid(tmp_path / "grid.safetensors", rows=4)

    with pytest.raises(ValueError, match="grid has 4 rows but this checkpoint's curve table has 3"):
        network.load_assistant_adapter(str(_adapter(tmp_path)), adaln_grid_path=str(grid_path))


def test_a_full_width_adaln_row_cannot_be_bridged_on_a_model_without_a_curve_table(tmp_path: Path) -> None:
    network = MiniMaxH3LoRANetwork(tiny_transformer(), rank=2, alpha=4.0, block_range=(1, 1))
    grid_path = _grid(tmp_path / "grid.safetensors")

    with pytest.raises(ValueError, match=r"adaln_proj.linear \(in 12 != 10\)"):
        network.load_assistant_adapter(str(_adapter(tmp_path)), adaln_grid_path=str(grid_path))


def test_only_adaln_rows_may_be_bridged_through_the_grid(tmp_path: Path) -> None:
    network = _pruned_network()
    path = write_adapter(tmp_path / "assist.safetensors", {"blocks.0.mlp.fc1": (*hand_pair(FULL_WIDTH, 16), 2.0)})
    grid_path = _grid(tmp_path / "grid.safetensors")

    with pytest.raises(ValueError, match=r"mlp.fc1 \(in 12 != 16\)"):
        network.load_assistant_adapter(str(path), adaln_grid_path=str(grid_path))
