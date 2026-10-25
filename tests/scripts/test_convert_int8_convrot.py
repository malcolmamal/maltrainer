"""scripts/convert_int8_convrot.py with a stand-in for comfy-kitchen's quantiser."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tests.scripts.convrot_world import CONVROT, blob, install_quantizer, make_files, run


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[torch.Tensor, int]]:
    return install_quantizer(monkeypatch)


def test_a_listed_linear_is_stored_as_int8_with_a_row_scale_and_the_reference_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[torch.Tensor, int]]
) -> None:
    source, reference, output = make_files(tmp_path)
    original = load_file(str(source))["blocks.0.lin.weight"].float()

    assert run(monkeypatch, source, output, reference) == 0

    out = load_file(str(output))
    weight, scale = calls[0][0], out["blocks.0.lin.weight_scale"]
    assert torch.equal(weight, original) and calls[0][1] == 4
    assert out["blocks.0.lin.weight"].dtype == torch.int8 and out["blocks.0.lin.weight"].shape == (3, 8)
    assert scale.dtype == torch.float32 and scale.shape == (3, 1)
    assert torch.equal(out["blocks.0.lin.comfy_quant"], blob(CONVROT))


def test_everything_else_is_copied_and_the_reference_blobs_are_added(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[torch.Tensor, int]]
) -> None:
    source, reference, output = make_files(tmp_path)
    plain = load_file(str(source))

    run(monkeypatch, source, output, reference)

    out = load_file(str(output))
    assert len(calls) == 1
    assert torch.equal(out["blocks.0.lin.bias"], plain["blocks.0.lin.bias"])
    assert torch.equal(out["blocks.0.norm.weight"], plain["blocks.0.norm.weight"])
    assert torch.equal(out["blocks.0.comfy_attention.config"], blob({"attention": "int8"}))
    assert "blocks.0.other_scale" not in out
    assert len(out) == 6 and not Path(f"{output}.tmp").exists()


@pytest.mark.usefixtures("calls")
def test_the_metadata_names_the_reference_and_the_source_and_keeps_the_sources_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source, reference, output = make_files(tmp_path)

    run(monkeypatch, source, output, reference)

    with safe_open(str(output), framework="pt") as handle:
        assert handle.metadata() == {
            "origin": "test",
            "int8_convrot_reference": "ref.safetensors",
            "int8_convrot_source": "src.safetensors",
        }
    summary = capsys.readouterr().out
    assert f"wrote {output} (" in summary
    assert "1 Linears INT8 ConvRot, 1 reference blobs, 6 tensors" in summary


def test_a_marker_without_a_group_size_means_256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[torch.Tensor, int]]
) -> None:
    source, reference, output = make_files(tmp_path, {"format": "int8_tensorwise", "convrot": True})

    run(monkeypatch, source, output, reference)

    assert calls[0][1] == 256


def test_a_source_tensor_wins_over_a_reference_blob_of_the_same_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[torch.Tensor, int]]
) -> None:
    source, reference, output = make_files(tmp_path)
    state = load_file(str(source))
    state["blocks.0.comfy_attention.config"] = blob("from the source")
    save_file(state, str(source))

    run(monkeypatch, source, output, reference)

    assert torch.equal(load_file(str(output))["blocks.0.comfy_attention.config"], blob("from the source"))
    assert calls


def test_a_source_saved_without_the_container_prefix_is_written_in_the_references_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calls: list[tuple[torch.Tensor, int]]
) -> None:
    # a Krea 2 finetune from krea2_raw_fp8_scaled keeps that file's bare names ("blocks.0..."), while the INT8
    # ConvRot releases ComfyUI loads say "model.diffusion_model.blocks.0..."
    source, plain_reference, output = make_files(tmp_path)
    reference = tmp_path / "prefixed_ref.safetensors"
    save_file({f"model.diffusion_model.{k}": v for k, v in load_file(str(plain_reference)).items()}, str(reference))
    plain = load_file(str(source))

    assert run(monkeypatch, source, output, reference) == 0

    out = load_file(str(output))
    assert sorted(out) == sorted(
        f"model.diffusion_model.blocks.0.{name}"
        for name in (
            "lin.weight",
            "lin.weight_scale",
            "lin.comfy_quant",
            "lin.bias",
            "norm.weight",
            "comfy_attention.config",
        )
    )
    assert out["model.diffusion_model.blocks.0.lin.weight"].dtype == torch.int8 and len(calls) == 1
    assert torch.equal(out["model.diffusion_model.blocks.0.norm.weight"], plain["blocks.0.norm.weight"])
