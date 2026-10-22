"""scripts/boost_text_fusion.py: scale the up-projections of a Krea 2 LoRA's four text-fusion blocks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import boost_text_fusion
import merge_loras

FUSION = "diffusion_model.txtfusion.refiner_blocks.0.attn.wq"
LAYERWISE = "diffusion_model.txtfusion.layerwise_blocks.1.attn.wk"
KOHYA_FUSION = "lora_unet_txtfusion_refiner_blocks_1_attn_wv"
DIT = "diffusion_model.blocks.3.attn.wq"
PROJECTOR = "diffusion_model.txtfusion.projector.proj"


def _lora(path: Path, names: list[str], kohya: bool = False, dtype: torch.dtype = torch.float32) -> Path:
    down, up = (".lora_down.weight", ".lora_up.weight") if kohya else (".lora_A.weight", ".lora_B.weight")
    gen = torch.Generator().manual_seed(1)
    state = {}
    for name in names:
        state[name + down] = torch.randn(2, 4, generator=gen).to(dtype)
        state[name + up] = torch.randn(3, 2, generator=gen).to(dtype)
        state[name + ".alpha"] = torch.tensor(2.0)
    save_file(state, str(path), metadata={"ss_network_dim": "2"})
    return path


def _run(monkeypatch: pytest.MonkeyPatch, *argv: object) -> int:
    monkeypatch.setattr("sys.argv", ["boost_text_fusion.py", *map(str, argv)])
    return boost_text_fusion.main()


def test_only_the_up_projections_inside_the_four_blocks_are_multiplied(tmp_path: Path) -> None:
    source = _lora(tmp_path / "a.safetensors", [FUSION, LAYERWISE, DIT, PROJECTOR])
    before = load_file(str(source))

    boost_text_fusion.boost_file(source, tmp_path / "out.safetensors", 3.0)

    after = load_file(str(tmp_path / "out.safetensors"))
    assert set(after) == set(before)
    boosted = {f"{FUSION}.lora_B.weight", f"{LAYERWISE}.lora_B.weight"}
    for key in before:
        expected = before[key] * 3.0 if key in boosted else before[key]
        assert torch.equal(after[key], expected), key


def test_kohya_spellings_with_underscores_are_found_too(tmp_path: Path) -> None:
    source = _lora(tmp_path / "k.safetensors", [KOHYA_FUSION, "lora_unet_blocks_3_attn_wq"], kohya=True)
    before = load_file(str(source))

    boost_text_fusion.boost_file(source, tmp_path / "out.safetensors", 2.0)

    after = load_file(str(tmp_path / "out.safetensors"))
    assert torch.equal(after[f"{KOHYA_FUSION}.lora_up.weight"], before[f"{KOHYA_FUSION}.lora_up.weight"] * 2)
    assert torch.equal(
        after["lora_unet_blocks_3_attn_wq.lora_up.weight"], before["lora_unet_blocks_3_attn_wq.lora_up.weight"]
    )
    assert torch.equal(after[f"{KOHYA_FUSION}.lora_down.weight"], before[f"{KOHYA_FUSION}.lora_down.weight"])


def test_the_tensor_dtype_is_kept_and_the_multiplication_is_done_in_float32(tmp_path: Path) -> None:
    source = _lora(tmp_path / "b.safetensors", [FUSION], dtype=torch.bfloat16)
    before = load_file(str(source))[f"{FUSION}.lora_B.weight"]

    boost_text_fusion.boost_file(source, tmp_path / "out.safetensors", 1.7)

    after = load_file(str(tmp_path / "out.safetensors"))[f"{FUSION}.lora_B.weight"]
    assert after.dtype == torch.bfloat16
    assert torch.equal(after, (before.float() * 1.7).to(torch.bfloat16))


def test_the_metadata_keeps_the_source_entries_and_records_the_boost(tmp_path: Path) -> None:
    source = _lora(tmp_path / "a.safetensors", [FUSION, LAYERWISE, KOHYA_FUSION])

    boost_text_fusion.boost_file(source, tmp_path / "out.safetensors", 2.5)

    with safe_open(str(tmp_path / "out.safetensors"), framework="pt") as handle:
        meta = handle.metadata()
    assert meta["ss_network_dim"] == "2"
    assert json.loads(meta["maltrainer_text_fusion_boost"]) == {
        "factor": 2.5,
        "blocks": ["txtfusion.layerwise_blocks.1", "txtfusion.refiner_blocks.0", "txtfusion.refiner_blocks.1"],
        "adapters": 3,
        "source": {"file": "a.safetensors", "sha256": merge_loras.sha256_prefix(source)},
    }


def test_a_lora_without_text_fusion_adapters_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    source = _lora(tmp_path / "a.safetensors", [DIT, PROJECTOR])

    with pytest.raises(ValueError, match=r"a\.safetensors: no text-fusion block adapters"):
        boost_text_fusion.boost_file(source, tmp_path / "sub" / "out.safetensors", 3.0)

    assert not (tmp_path / "sub").exists()


def test_the_output_folder_is_created_and_the_result_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _lora(tmp_path / "a.safetensors", [FUSION, LAYERWISE])
    target = tmp_path / "deep" / "er" / "out.safetensors"

    boost_text_fusion.boost_file(source, target, 3.0)

    assert target.is_file()
    assert capsys.readouterr().out.splitlines() == [
        "  x3 on 2 adapters in 2 blocks: txtfusion.layerwise_blocks.1, txtfusion.refiner_blocks.0",
        f"  wrote {target}",
    ]
