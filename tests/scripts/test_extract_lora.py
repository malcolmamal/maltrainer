from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import extract_lora

OUT, IN = 5, 7
GENERATOR = torch.Generator().manual_seed(1)
BASE_W = torch.randn(OUT, IN, generator=GENERATOR)
# The change is exactly rank 2: 3 * u1 v1^T + 1 * u2 v2^T with orthonormal pairs.
Q_U = torch.linalg.qr(torch.randn(OUT, 2, generator=GENERATOR))[0]
Q_V = torch.linalg.qr(torch.randn(IN, 2, generator=GENERATOR))[0]
CHANGE = Q_U @ torch.diag(torch.tensor([3.0, 1.0])) @ Q_V.T


@pytest.fixture
def checkpoints(tmp_path: Path):
    base = {"layer.weight": BASE_W, "other.weight": BASE_W * 2, "layer.bias": torch.zeros(OUT)}
    tuned = {**base, "layer.weight": BASE_W + CHANGE, "layer.bias": torch.ones(OUT)}
    save_file(base, str(tmp_path / "base.safetensors"))
    save_file(tuned, str(tmp_path / "ft.safetensors"))
    return tmp_path / "base.safetensors", tmp_path / "ft.safetensors"


def run(monkeypatch, *argv) -> int:
    monkeypatch.setattr("sys.argv", ["extract_lora.py", *map(str, argv), "--device", "cpu", "--dtype", "float32"])
    return extract_lora.main()


def test_main_writes_one_file_per_rank_that_reproduces_the_change(checkpoints, monkeypatch, capsys):
    base, tuned = checkpoints
    assert run(monkeypatch, base, tuned, "--rank", "4", "1") == 0
    out = capsys.readouterr().out
    assert "1 tensors unchanged" in out  # other.weight

    full = load_file(str(tuned.parent / "ft_lora_r4.safetensors"))
    assert sorted(full) == [
        "diffusion_model.layer.alpha",
        "diffusion_model.layer.lora_A.weight",
        "diffusion_model.layer.lora_B.weight",
    ]
    assert full["diffusion_model.layer.lora_A.weight"].shape == (4, IN)  # fixed mode: the cap
    assert float(full["diffusion_model.layer.alpha"]) == 4.0
    got = full["diffusion_model.layer.lora_B.weight"] @ full["diffusion_model.layer.lora_A.weight"]
    assert torch.allclose(got, CHANGE, atol=1e-5)

    top = load_file(str(tuned.parent / "ft_lora_r1.safetensors"))
    got1 = top["diffusion_model.layer.lora_B.weight"] @ top["diffusion_model.layer.lora_A.weight"]
    assert torch.allclose(got1, 3.0 * Q_U[:, :1] @ Q_V[:, :1].T, atol=1e-5)
    assert torch.linalg.matrix_rank(got1) == 1


def test_main_metadata(checkpoints, monkeypatch):
    base, tuned = checkpoints
    assert run(monkeypatch, base, tuned, "--rank", "8", "--name", "mine", "--output-dir", tuned.parent / "o") == 0
    with safe_open(str(tuned.parent / "o" / "mine_r8.safetensors"), framework="pt") as handle:
        meta = handle.metadata()
    assert meta["ss_network_dim"] == "5"  # min(cap, out, in): rank can't exceed the layer
    recipe = json.loads(meta["maltrainer_extract"])
    assert recipe["rank_cap"] == 8 and recipe["mode"] == "fixed" and recipe["profile"] == "generic"


def test_fro_mode_keeps_enough_rank_for_the_share(checkpoints):
    base, tuned = checkpoints
    # 3^2 / (3^2 + 1^2) = 0.9
    for value, rank in ((0.85, 1), (0.95, 2)):
        states, report = extract_lora.extract(base, tuned, [4], mode="fro", value=value, device="cpu")
        assert report["per_cap"][4]["rank_max"] == rank
        assert states[4]["diffusion_model.layer.lora_A.weight"].shape[0] == rank


def test_dropped_share_is_reported(checkpoints):
    base, tuned = checkpoints
    _, report = extract_lora.extract(base, tuned, [1], device="cpu")
    assert report["per_cap"][1]["dropped_worst"] == pytest.approx((1 / 10) ** 0.5, rel=1e-4)


def test_shape_and_key_checks(checkpoints, tmp_path):
    base, _tuned = checkpoints
    save_file(
        {"layer.weight": torch.zeros(OUT + 1, IN), "other.weight": BASE_W, "layer.bias": torch.zeros(OUT)},
        str(tmp_path / "bad_shape.safetensors"),
    )
    with pytest.raises(ValueError, match="differ in shape"):
        extract_lora.extract(base, tmp_path / "bad_shape.safetensors", [2], device="cpu")
    save_file({"layer.weight": BASE_W}, str(tmp_path / "bad_keys.safetensors"))
    with pytest.raises(ValueError, match="different tensors"):
        extract_lora.extract(base, tmp_path / "bad_keys.safetensors", [2], device="cpu")
    with pytest.raises(ValueError, match="no LoRA target changed"):
        extract_lora.extract(base, base, [2], device="cpu")


def test_main_reports_errors_as_exit_code_1(checkpoints, monkeypatch, capsys):
    base, _ = checkpoints
    assert run(monkeypatch, base, base) == 1
    assert "no LoRA target changed" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--rank", "0"], "--rank must be at least 1"),
        (["--mode", "ratio"], "needs --value"),
        (["--mode", "fro", "--value", "1.5"], "(0, 1]"),
        ([], "pass --force"),
    ],
)
def test_main_argument_errors(checkpoints, monkeypatch, capsys, extra, message):
    base, tuned = checkpoints
    (tuned.parent / "ft_lora_r64.safetensors").write_bytes(b"x")
    with pytest.raises(SystemExit) as raised:
        run(monkeypatch, base, tuned, *extra)
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_main_missing_file(checkpoints, monkeypatch, capsys):
    base, tuned = checkpoints
    with pytest.raises(SystemExit):
        run(monkeypatch, base, tuned.parent / "nope.safetensors")
    assert "not found" in capsys.readouterr().err


def test_detect_profile_falls_back_to_generic(checkpoints):
    assert extract_lora.detect_profile(checkpoints[0]).name == "generic"


def test_changed_tensors_that_are_not_lora_targets_are_reported_and_left_out(tmp_path, monkeypatch, capsys):
    base = {"layer.weight": BASE_W, "norm.weight": torch.ones(OUT)}
    tuned = {"layer.weight": BASE_W + CHANGE, "norm.weight": torch.full((OUT,), 2.0)}
    save_file(base, str(tmp_path / "base.safetensors"))
    save_file(tuned, str(tmp_path / "ft.safetensors"))

    assert run(monkeypatch, tmp_path / "base.safetensors", tmp_path / "ft.safetensors", "--rank", "2") == 0

    assert "1 changed tensors are not LoRA targets and are left out, e.g. ['norm']" in capsys.readouterr().out


def test_a_moss_image_checkpoint_is_recognised_by_its_layout(tmp_path):
    path = tmp_path / "moss.safetensors"
    save_file(
        {"shared_block_modulation.weight": torch.zeros(2, 2), "layers.0.proj.ffn_up_gate.weight": torch.zeros(2, 2)},
        str(path),
    )

    assert extract_lora.detect_profile(path).name == "moss_image"
