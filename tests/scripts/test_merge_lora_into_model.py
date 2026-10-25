"""scripts/merge_lora_into_model.py: LoRAs and full deltas baked into a base, at their strengths, exactly."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import merge_lora_into_model

P = "model.diffusion_model."
GEN = torch.Generator().manual_seed(4)
W = torch.randn(4, 6, generator=GEN)
BIAS = torch.randn(4, generator=GEN)
NORM = torch.randn(6, generator=GEN)


def _base(path: Path, dtype: torch.dtype = torch.float32, extra: dict[str, torch.Tensor] | None = None) -> Path:
    tensors = {
        f"{P}blocks.0.lin.weight": W.to(dtype),
        f"{P}blocks.0.lin.bias": BIAS.to(dtype),
        f"{P}blocks.0.norm.scale": NORM.to(dtype),
        f"{P}step": torch.tensor([3]),
        **(extra or {}),
    }
    save_file(tensors, str(path), metadata={"origin": "base"})
    return path


def _lora(path: Path, down: torch.Tensor, up: torch.Tensor, alpha: float | None = None, kohya: bool = False) -> Path:
    names = ("lora_down", "lora_up") if kohya else ("lora_A", "lora_B")
    tensors = {
        f"diffusion_model.blocks.0.lin.{names[0]}.weight": down,
        f"diffusion_model.blocks.0.lin.{names[1]}.weight": up,
    }
    if alpha is not None:
        tensors["diffusion_model.blocks.0.lin.alpha"] = torch.tensor(alpha)
    save_file(tensors, str(path))
    return path


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr("sys.argv", ["merge", *args, "--device", "cpu"])
    return merge_lora_into_model.main()


def test_two_loras_add_strength_times_alpha_over_rank_times_b_a_each_and_the_rest_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, out = _base(tmp_path / "base.safetensors"), tmp_path / "out.safetensors"
    a1, b1 = torch.randn(2, 6, generator=GEN), torch.randn(4, 2, generator=GEN)
    a2, b2 = torch.randn(3, 6, generator=GEN), torch.randn(4, 3, generator=GEN)
    first = _lora(tmp_path / "a.safetensors", a1, b1, alpha=1.0)
    second = _lora(tmp_path / "b.safetensors", a2, b2, kohya=True)  # no alpha: scale 1

    assert _run(monkeypatch, str(base), str(out), f"{first}:0.5", f"{second}:2") == 0

    merged = load_file(str(out))
    expected = W + 0.5 * (1.0 / 2) * (b1 @ a1) + 2.0 * (b2 @ a2)
    torch.testing.assert_close(merged[f"{P}blocks.0.lin.weight"], expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(merged[f"{P}blocks.0.lin.bias"], BIAS) and torch.equal(merged[f"{P}blocks.0.norm.scale"], NORM)
    assert torch.equal(merged[f"{P}step"], torch.tensor([3]))


def test_comfyui_full_deltas_add_to_the_weight_and_the_bias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base, out = _base(tmp_path / "base.safetensors"), tmp_path / "out.safetensors"
    diff, diff_b = torch.randn(4, 6, generator=GEN), torch.randn(4, generator=GEN)
    full = tmp_path / "full.safetensors"
    save_file({"diffusion_model.blocks.0.lin.diff": diff, "diffusion_model.blocks.0.lin.diff_b": diff_b}, str(full))

    assert _run(monkeypatch, str(base), str(out), f"{full}:0.25") == 0

    merged = load_file(str(out))
    torch.testing.assert_close(merged[f"{P}blocks.0.lin.weight"], W + 0.25 * diff)
    torch.testing.assert_close(merged[f"{P}blocks.0.lin.bias"], BIAS + 0.25 * diff_b)


def test_the_output_keeps_the_bases_dtype_unless_asked_and_records_the_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, out = _base(tmp_path / "base.safetensors", torch.float16), tmp_path / "out.safetensors"
    lora = _lora(tmp_path / "a.safetensors", torch.ones(1, 6), torch.ones(4, 1))

    assert _run(monkeypatch, str(base), str(out), str(lora)) == 0
    assert {t.dtype for t in load_file(str(out)).values() if t.is_floating_point()} == {torch.float16}
    with safe_open(str(out), framework="pt") as handle:
        meta = handle.metadata()
    recipe = json.loads(meta["maltrainer_merge_into_model"])
    assert (
        meta["origin"] == "base" and recipe["base"]["file"] == "base.safetensors" and recipe["save_dtype"] == "float16"
    )
    assert len(recipe["base"]["sha256"]) == 12 and recipe["models"] == [] and recipe["add_difference"] == []
    assert [(i["file"], i["strength"], i["modules"], len(i["sha256"])) for i in recipe["loras"]] == [
        ("a.safetensors", 1.0, 1, 12)
    ]

    assert _run(monkeypatch, str(base), str(out), str(lora), "--force", "--dtype", "bfloat16") == 0
    assert load_file(str(out))[f"{P}blocks.0.lin.weight"].dtype == torch.bfloat16


def test_an_fp8_scaled_base_is_dequantized_into_a_plain_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scale = torch.tensor([[0.5], [0.25], [1.0], [2.0]])
    fp8 = (W / scale).to(torch.float8_e4m3fn)
    base = tmp_path / "fp8.safetensors"
    tensors = {f"{P}blocks.0.lin.weight": fp8, f"{P}blocks.0.lin.weight_scale": scale, f"{P}blocks.0.lin.bias": BIAS}
    save_file(tensors, str(base), metadata={"_quantization_metadata": "{}", "origin": "fp8"})
    lora, out = _lora(tmp_path / "a.safetensors", torch.ones(1, 6), torch.ones(4, 1)), tmp_path / "out.safetensors"

    assert _run(monkeypatch, str(base), str(out), str(lora)) == 0

    merged = load_file(str(out))
    assert sorted(merged) == [f"{P}blocks.0.lin.bias", f"{P}blocks.0.lin.weight"]
    expected = (fp8.float() * scale + torch.ones(4, 6)).to(torch.bfloat16)  # an fp8 base writes bf16 by default
    assert torch.equal(merged[f"{P}blocks.0.lin.weight"], expected)
    with safe_open(str(out), framework="pt") as handle:
        assert "_quantization_metadata" not in handle.metadata() and handle.metadata()["origin"] == "fp8"


@pytest.mark.parametrize(
    "case, message",
    [
        ("int8_convrot", r"base\.safetensors is INT8 ConvRot: run scripts/dequantize_int8_convrot\.py"),
        ("int8_scaled", r"blocks\.0\.lin\.weight_scale scales a I8 weight"),
        ("missing_module", r"1 LoRA modules are not in base\.safetensors, e\.g\. \['blocks\.9\.lin'\]"),
        ("wrong_shape", r"blocks\.0\.lin\.weight: the delta is \(5, 6\), the base weight \(4, 6\)"),
        ("dora", r"a\.safetensors: tensors this merge does not understand"),
    ],
)
def test_what_it_cannot_merge_exactly_is_refused_and_nothing_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str, message: str
) -> None:
    base, lora = tmp_path / "base.safetensors", tmp_path / "a.safetensors"
    if case == "int8_convrot":
        _base(base, extra={f"{P}blocks.0.lin.comfy_quant": torch.zeros(2, dtype=torch.uint8)})
    elif case == "int8_scaled":
        save_file(
            {
                f"{P}blocks.0.lin.weight": torch.ones(4, 6, dtype=torch.int8),
                f"{P}blocks.0.lin.weight_scale": torch.ones(4, 1),
            },
            str(base),
        )
    else:
        _base(base)
    if case == "missing_module":
        save_file(
            {
                "diffusion_model.blocks.9.lin.lora_A.weight": torch.ones(1, 6),
                "diffusion_model.blocks.9.lin.lora_B.weight": torch.ones(4, 1),
            },
            str(lora),
        )
    elif case == "wrong_shape":
        _lora(lora, torch.ones(1, 6), torch.ones(5, 1))
    elif case == "dora":
        save_file(
            {
                "diffusion_model.blocks.0.lin.lora_A.weight": torch.ones(1, 6),
                "diffusion_model.blocks.0.lin.lora_B.weight": torch.ones(4, 1),
                "diffusion_model.blocks.0.lin.dora_scale": torch.ones(4),
            },
            str(lora),
        )
    else:
        _lora(lora, torch.ones(1, 6), torch.ones(4, 1))
    out = tmp_path / "out.safetensors"

    assert _run(monkeypatch, str(base), str(out), str(lora)) == 1
    assert re.search(message, capsys.readouterr().err) and not out.exists()


def test_a_missing_input_or_an_existing_output_stops_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base, out = _base(tmp_path / "base.safetensors"), tmp_path / "out.safetensors"
    with pytest.raises(SystemExit):
        _run(monkeypatch, str(base), str(out), str(tmp_path / "nope.safetensors"))
    assert "not found:" in capsys.readouterr().err
    out.write_bytes(b"keep")
    with pytest.raises(SystemExit):
        _run(
            monkeypatch, str(base), str(out), str(_lora(tmp_path / "a.safetensors", torch.ones(1, 6), torch.ones(4, 1)))
        )
    assert out.read_bytes() == b"keep" and "pass --force" in capsys.readouterr().err
