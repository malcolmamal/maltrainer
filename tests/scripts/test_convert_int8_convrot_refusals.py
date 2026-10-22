"""scripts/convert_int8_convrot.py: progress output and the ways a conversion is refused."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.scripts.convrot_world import QUANT, install_quantizer, make_files, run


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, int]]:
    return install_quantizer(monkeypatch)


def test_progress_is_printed_every_fifty_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[torch.Tensor, int]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference, output = make_files(tmp_path)
    state = load_file(str(source)) | {f"extra.{i}": torch.zeros(1) for i in range(100)}
    save_file(state, str(source))

    run(monkeypatch, source, output, reference)

    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["  50/103 tensors", "  100/103 tensors"]
    assert calls


def test_without_comfy_kitchen_it_asks_for_comfyuis_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source, reference, output = make_files(tmp_path)
    monkeypatch.setitem(sys.modules, QUANT, None)

    assert run(monkeypatch, source, output, reference) == 1

    assert "comfy_kitchen is missing: run this with ComfyUI's Python" in capsys.readouterr().err
    assert not output.exists()


def test_an_existing_output_is_kept_unless_force_is_given(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[torch.Tensor, int]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference, output = make_files(tmp_path)
    output.write_bytes(b"old")

    assert run(monkeypatch, source, output, reference) == 1
    assert output.read_bytes() == b"old" and f"{output} exists (pass --force to overwrite)" in capsys.readouterr().err
    assert run(monkeypatch, source, output, reference, "--force") == 0

    assert "blocks.0.lin.weight_scale" in load_file(str(output)) and len(calls) == 1


def test_a_reference_linear_missing_from_the_source_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[torch.Tensor, int]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, reference, output = make_files(tmp_path)
    save_file({"blocks.0.norm.weight": torch.zeros(8)}, str(source))

    assert run(monkeypatch, source, output, reference) == 1

    assert (
        "the reference quantizes 1 Linears the source does not have, e.g. ['blocks.0.lin']" in capsys.readouterr().err
    )
    assert not calls and not output.exists()


@pytest.mark.parametrize("config", [{"format": "fp8"}, {"format": "int8_tensorwise", "convrot": False}])
def test_a_reference_that_is_not_int8_convrot_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[torch.Tensor, int]],
    capsys: pytest.CaptureFixture[str],
    config: dict[str, Any],
) -> None:
    source, reference, output = make_files(tmp_path, config)

    assert run(monkeypatch, source, output, reference) == 1

    assert f"blocks.0.lin: reference format {config} is not INT8 ConvRot" in capsys.readouterr().err
    assert not calls and not output.exists()
