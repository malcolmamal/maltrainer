from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.minimax_h3.int8_convrot import load_comfy_int8_convrot_state_dict

CPU = torch.device("cpu")
MARKER = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}


def _blob(config: dict | str) -> torch.Tensor:
    payload = config if isinstance(config, str) else json.dumps(config)
    return torch.tensor(list(payload.encode()), dtype=torch.uint8)


def _layer(prefix: str, marker: dict | str = MARKER, rows: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(len(prefix))
    return {
        f"{prefix}.weight": torch.randint(-127, 128, (rows, 32), dtype=torch.int8, generator=generator),
        f"{prefix}.weight_scale": torch.rand(rows, 1, generator=generator),
        f"{prefix}.comfy_quant": _blob(marker),
    }


def _write(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> str:
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    return str(path)


def test_a_quantized_layer_loads_as_weight_scale_weight_and_group_size_buffers(tmp_path: Path) -> None:
    tensors = {**_layer("blk.0"), "blk.0.bias": torch.ones(4), "norm.weight": torch.full((3,), 2.0)}

    state, count = load_comfy_int8_convrot_state_dict(_write(tmp_path, tensors), device=CPU)

    assert count == 1
    assert sorted(state) == [
        "blk.0.bias",
        "blk.0.int8_convrot_groupsize",
        "blk.0.scale_weight",
        "blk.0.weight",
        "norm.weight",
    ]
    assert torch.equal(state["blk.0.weight"], tensors["blk.0.weight"])
    assert state["blk.0.int8_convrot_groupsize"].dtype == torch.int32
    assert state["blk.0.int8_convrot_groupsize"].item() == 16
    torch.testing.assert_close(state["norm.weight"], tensors["norm.weight"])


def test_scales_are_widened_to_float32_columns_whatever_their_stored_shape(tmp_path: Path) -> None:
    tensors = _layer("a")
    tensors["a.weight_scale"] = tensors["a.weight_scale"].reshape(-1).half()

    state, _ = load_comfy_int8_convrot_state_dict(_write(tmp_path, tensors), device=CPU)

    assert state["a.scale_weight"].dtype == torch.float32
    assert state["a.scale_weight"].shape == (4, 1)
    torch.testing.assert_close(state["a.scale_weight"], tensors["a.weight_scale"].float().reshape(-1, 1))


def test_every_layer_keeps_its_own_group_size(tmp_path: Path) -> None:
    tensors = {**_layer("a"), **_layer("b", {**MARKER, "convrot_groupsize": 4})}

    state, count = load_comfy_int8_convrot_state_dict(_write(tmp_path, tensors), device=CPU)

    assert count == 2
    assert (state["a.int8_convrot_groupsize"].item(), state["b.int8_convrot_groupsize"].item()) == (16, 4)


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        ({"format": "nvfp4", "convrot": True, "convrot_groupsize": 16}, "is not an INT8 ConvRot marker"),
        ({"format": "int8_tensorwise", "convrot": False, "convrot_groupsize": 16}, "is not an INT8 ConvRot marker"),
        ({"format": "int8_tensorwise", "convrot": True}, "invalid ConvRot group size 0"),
        ({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 8}, "invalid ConvRot group size 8"),
        ("{broken", "Invalid Comfy quantization marker"),
    ],
)
def test_a_marker_that_does_not_describe_int8_convrot_is_rejected(tmp_path: Path, marker: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_comfy_int8_convrot_state_dict(_write(tmp_path, _layer("a", marker)), device=CPU)


def test_a_file_without_markers_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="inconsistent marker, scale, and quantized-weight sets: markers=0"):
        load_comfy_int8_convrot_state_dict(_write(tmp_path, {"w.weight": torch.ones(2, 2)}), device=CPU)


def _without(tensors: dict[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    return {name: value for name, value in tensors.items() if name != key}


@pytest.mark.parametrize(
    ("damage", "counts"),
    [
        (lambda t: _without(t, "a.weight_scale"), "markers=2, scales=1, weights=2"),
        (lambda t: _without(t, "a.comfy_quant"), "markers=1, scales=2, weights=2"),
        (lambda t: {**t, "a.weight": t["a.weight"].float()}, "markers=2, scales=2, weights=1"),
        (lambda t: {**t, "c.weight": torch.zeros(2, 2, dtype=torch.int8)}, "markers=2, scales=2, weights=3"),
    ],
)
def test_markers_scales_and_int8_weights_must_describe_the_same_layers(tmp_path: Path, damage, counts: str) -> None:
    tensors = damage({**_layer("a"), **_layer("b")})

    with pytest.raises(ValueError, match=counts):
        load_comfy_int8_convrot_state_dict(_write(tmp_path, tensors), device=CPU)
