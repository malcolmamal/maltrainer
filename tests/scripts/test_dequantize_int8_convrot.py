"""scripts/dequantize_int8_convrot.py with a stand-in for comfy-kitchen's INT8 ConvRot inverse."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import dequantize_int8_convrot
from tests.scripts.convrot_world import CONVROT, QUANT, blob


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    """comfy-kitchen's module with an inverse that scales, then 'rotates back' by negating: visibly not just q*s."""
    seen: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def dequantize(q: torch.Tensor, scale: torch.Tensor, group: int) -> torch.Tensor:
        seen.append((q, scale, group))
        return -(q.float() * scale.float())

    for name in ("comfy_kitchen", "comfy_kitchen.backends", "comfy_kitchen.backends.eager", QUANT):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    vars(sys.modules[QUANT])["dequantize_int8_convrot_weight"] = dequantize
    return seen


def _int8_file(path: Path, marker: dict[str, object] | str = CONVROT) -> Path:
    save_file(
        {
            "model.diffusion_model.blocks.0.lin.weight": torch.tensor([[1, -2], [3, 4], [-5, 6]], dtype=torch.int8),
            "model.diffusion_model.blocks.0.lin.weight_scale": torch.tensor([[0.5], [0.25], [2.0]]),
            "model.diffusion_model.blocks.0.lin.comfy_quant": blob(marker),
            "model.diffusion_model.blocks.0.lin.bias": torch.tensor([0.1, 0.2, 0.3], dtype=torch.bfloat16),
            "model.diffusion_model.blocks.0.norm.scale": torch.tensor([1.5, 2.5], dtype=torch.bfloat16),
            "model.diffusion_model.step": torch.tensor([7], dtype=torch.int64),
        },
        str(path),
        metadata={"origin": "test"},
    )
    return path


def _run(monkeypatch: pytest.MonkeyPatch, source: Path, output: Path, *flags: str) -> int:
    monkeypatch.setattr("sys.argv", ["dequantize", str(source), str(output), "--device", "cpu", *flags])
    return dequantize_int8_convrot.main()


def test_a_marked_linear_is_scaled_and_rotated_back_and_the_markers_are_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[torch.Tensor, torch.Tensor, int]]
) -> None:
    source, output = _int8_file(tmp_path / "int8.safetensors"), tmp_path / "out.safetensors"

    assert _run(monkeypatch, source, output) == 0

    out = load_file(str(output))
    assert sorted(out) == [
        "model.diffusion_model.blocks.0.lin.bias",
        "model.diffusion_model.blocks.0.lin.weight",
        "model.diffusion_model.blocks.0.norm.scale",
        "model.diffusion_model.step",
    ]
    q, scale, group = calls[0]
    assert q.dtype == torch.int8 and scale.shape == (3, 1) and group == 4
    expected = -(torch.tensor([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]]) * torch.tensor([[0.5], [0.25], [2.0]]))
    assert torch.equal(out["model.diffusion_model.blocks.0.lin.weight"], expected.to(torch.float16))
    assert out["model.diffusion_model.blocks.0.norm.scale"].dtype == torch.float16
    assert torch.equal(out["model.diffusion_model.step"], torch.tensor([7]))  # not floating: kept as stored


@pytest.mark.usefixtures("calls")
def test_bfloat16_is_written_on_request_and_the_metadata_names_the_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output = _int8_file(tmp_path / "int8.safetensors"), tmp_path / "out.safetensors"

    assert _run(monkeypatch, source, output, "--dtype", "bfloat16") == 0

    assert {t.dtype for t in load_file(str(output)).values() if t.is_floating_point()} == {torch.bfloat16}
    with safe_open(str(output), framework="pt") as handle:
        assert handle.metadata() == {"origin": "test", "int8_convrot_dequantized_from": "int8.safetensors"}


def test_a_marker_without_a_group_size_means_256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[torch.Tensor, torch.Tensor, int]]
) -> None:
    source = _int8_file(tmp_path / "int8.safetensors", {"format": "int8_tensorwise", "convrot": True})
    assert _run(monkeypatch, source, tmp_path / "out.safetensors") == 0
    assert calls[0][2] == 256


@pytest.mark.parametrize("marker", [{"format": "float8_e4m3fn"}, {"format": "int8_tensorwise", "convrot": False}])
@pytest.mark.usefixtures("calls")
def test_any_other_quantization_is_refused_and_nothing_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    marker: dict[str, object],
) -> None:
    output = tmp_path / "out.safetensors"
    assert _run(monkeypatch, _int8_file(tmp_path / "int8.safetensors", marker), output) == 1
    err = capsys.readouterr().err
    assert "model.diffusion_model.blocks.0.lin:" in err and "is not INT8 ConvRot" in err and not output.exists()


@pytest.mark.usefixtures("calls")
def test_a_file_without_markers_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plain = tmp_path / "plain.safetensors"
    save_file({"blocks.0.lin.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, str(plain))

    assert _run(monkeypatch, plain, tmp_path / "out.safetensors") == 1
    assert "plain.safetensors has no INT8 ConvRot Linears" in capsys.readouterr().err


@pytest.mark.usefixtures("calls")
def test_an_existing_output_needs_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, output = _int8_file(tmp_path / "int8.safetensors"), tmp_path / "out.safetensors"
    output.write_bytes(b"keep me")

    assert _run(monkeypatch, source, output) == 1
    assert output.read_bytes() == b"keep me" and "pass --force" in capsys.readouterr().err
    assert _run(monkeypatch, source, output, "--force") == 0


def test_without_comfy_kitchen_it_says_which_python_to_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "comfy_kitchen", None)
    assert _run(monkeypatch, _int8_file(tmp_path / "int8.safetensors"), tmp_path / "out.safetensors") == 1
    assert "comfy_kitchen is missing: run this with ComfyUI's Python" in capsys.readouterr().err
