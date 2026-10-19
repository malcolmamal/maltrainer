from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import merge_loras
from tests.scripts.helpers import KOHYA, delta, write_lora

D1 = torch.tensor([[1.0, 0, 2, 0], [0, 1, 0, 3]])  # rank 2, in 4
U1 = torch.tensor([[1.0, 0], [0, 2], [1, 1]])  # out 3
D2 = torch.tensor([[0.0, 1, 1, 1]])  # rank 1
U2 = torch.tensor([[1.0], [-1], [2]])


def merged_delta(path: Path) -> tuple[torch.Tensor, float]:
    state = load_file(str(path))
    return state["m.lora_B.weight"].double() @ state["m.lora_A.weight"].double(), float(state["m.alpha"])


def run(monkeypatch, *argv) -> int:
    monkeypatch.setattr("sys.argv", ["merge_loras.py", *map(str, argv)])
    return merge_loras.main()


@pytest.mark.parametrize(
    ("spec", "path", "strength"),
    [
        ("a.safetensors", "a.safetensors", 1.0),
        ("a.safetensors:0.8", "a.safetensors", 0.8),
        ("C:\\x\\a.safetensors", "C:\\x\\a.safetensors", 1.0),
        ("C:/x/a.safetensors:-1.5", "C:/x/a.safetensors", -1.5),
        ("a.safetensors:high", "a.safetensors:high", 1.0),
    ],
)
def test_parse_input(spec, path, strength):
    assert merge_loras.parse_input(spec) == (Path(path), strength)


def test_merge_bakes_strength_and_alpha_and_sums_ranks(tmp_path, monkeypatch):
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, 4.0)})  # scale 4/2 = 2
    b = write_lora(tmp_path / "b.safetensors", {"m": (D2, U2, None)})  # no alpha -> scale 1
    out = tmp_path / "out.safetensors"
    assert run(monkeypatch, out, f"{a}:0.5", f"{b}:3", "--verify") == 0

    merged, alpha = merged_delta(out)
    expected = 0.5 * delta(D1, U1, 4.0) + 3.0 * delta(D2, U2)
    assert torch.allclose(merged, expected, atol=1e-6)
    assert alpha == 3.0  # rank 2 + rank 1, so scale 1
    assert load_file(str(out))["m.lora_A.weight"].shape == (3, 4)
    with safe_open(str(out), framework="pt") as handle:
        meta = handle.metadata()
    assert meta["ss_network_dim"] == "3"
    recipe = json.loads(meta["maltrainer_merge"])
    assert [i["strength"] for i in recipe["inputs"]] == [0.5, 3.0]
    assert recipe["max_rel_error"] < 1e-6


def test_single_lora_bakes_a_strength(tmp_path):
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, 1.0)})
    state, info, _ = merge_loras.merge([(a, 2.0)], torch.float32, verify=True)
    assert info["ranks"] == [2]
    merged = state["m.lora_B.weight"].double() @ state["m.lora_A.weight"].double()
    assert torch.allclose(merged, 2.0 * delta(D1, U1, 1.0), atol=1e-6)


def test_kohya_layout_is_kept_and_partial_modules_merge(tmp_path):
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, None), "n": (D2, U2, None)}, layout=KOHYA)
    b = write_lora(tmp_path / "b.safetensors", {"m": (D2, U2, None)}, layout=KOHYA)
    state, info, _ = merge_loras.merge([(a, 1.0), (b, 1.0)], torch.float32, verify=False)
    assert sorted(state) == [
        "m.alpha",
        "m.lora_down.weight",
        "m.lora_up.weight",
        "n.alpha",
        "n.lora_down.weight",
        "n.lora_up.weight",
    ]
    assert info["partial_modules"] == 1
    assert state["m.lora_down.weight"].shape[0] == 3
    assert state["n.lora_down.weight"].shape[0] == 1  # only in a: its own rank


def test_bfloat16_output_dtype(tmp_path):
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, None)})
    state, _, _ = merge_loras.merge([(a, 1.0)], torch.bfloat16, verify=False)
    assert state["m.lora_A.weight"].dtype == torch.bfloat16
    assert state["m.alpha"].dtype == torch.float32


def test_refusals(tmp_path):
    other = write_lora(tmp_path / "o.safetensors", {"m": (torch.ones(1, 5), U2, None)})
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, None)})
    with pytest.raises(ValueError, match="disagree on the layer shape"):
        merge_loras.merge([(a, 1.0), (other, 1.0)], torch.float32, verify=False)

    odd = tmp_path / "dora.safetensors"
    save_file({"m.lora_A.weight": D1, "m.lora_B.weight": U1, "m.dora_scale": torch.ones(3)}, str(odd))
    with pytest.raises(ValueError, match="does not understand"):
        merge_loras.read_lora(odd)
    save_file({"x.weight": torch.ones(2, 2)}, str(odd))
    with pytest.raises(ValueError, match="no lora_A"):
        merge_loras.read_lora(odd)


def test_load_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    body = {
        "output": "o.st",
        "dtype": "bfloat16",
        "verify": True,
        "loras": ["a.st", {"path": "b.st", "strength": 0.25}],
    }
    cfg.write_text(yaml.dump(body))
    assert merge_loras.load_config(cfg) == {
        "output": "o.st",
        "dtype": "bfloat16",
        "verify": True,
        "inputs": [(Path("a.st"), 1.0), (Path("b.st"), 0.25)],
    }


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"loras": []},
        {"loras": [{"strength": 1}]},
        {"loras": ["a"], "bogus": 1},
        {"loras": [{"path": "a", "x": 1}]},
        {"loras": ["a"], "dtype": "int4"},
    ],
)
def test_load_config_rejects(tmp_path, body):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.dump(body))
    with pytest.raises(ValueError, match=re.escape("c.yaml")):
        merge_loras.load_config(cfg)


def test_main_config_and_overrides(tmp_path, monkeypatch):
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, None)})
    out = tmp_path / "sub" / "out.safetensors"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.dump({"output": str(out), "loras": [{"path": str(a), "strength": 2}]}))
    assert run(monkeypatch, "--config", cfg, "--dtype", "float16") == 0
    assert load_file(str(out))["m.lora_A.weight"].dtype == torch.float16
    merged, _ = merged_delta(out)
    assert torch.allclose(merged, 2 * delta(D1, U1), atol=1e-2)


def test_main_argument_errors(tmp_path, monkeypatch, capsys):
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, None)})
    out = tmp_path / "out.safetensors"
    out.write_bytes(b"x")
    cases = [
        ([out], "give an output"),
        ([out, tmp_path / "missing.safetensors"], "not found"),
        ([out, a], "pass --force"),
        (["--config", tmp_path / "nope.yaml", "--force"], "nope.yaml"),
    ]
    for argv, message in cases:
        with pytest.raises(SystemExit) as raised:
            run(monkeypatch, *argv)
        assert raised.value.code == 2
        assert message in capsys.readouterr().err
    assert run(monkeypatch, out, a, "--force") == 0
    assert out.stat().st_size > 1


def test_main_returns_1_for_incompatible_inputs(tmp_path, monkeypatch, capsys):
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, None)})
    b = write_lora(tmp_path / "b.safetensors", {"m": (torch.ones(1, 5), U2, None)})
    assert run(monkeypatch, tmp_path / "o.safetensors", a, b) == 1
    assert "disagree" in capsys.readouterr().err
    assert not (tmp_path / "o.safetensors").exists()


def test_metadata_keeps_shared_identity_only(tmp_path):
    shared = {"ss_base_model_version": "base1", "modelspec.architecture": "arch"}
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, None)}, metadata={**shared, "ss_network_module": "x"})
    b = write_lora(tmp_path / "b.safetensors", {"m": (D2, U2, None)}, metadata={**shared, "ss_network_module": "y"})
    _, info, loras = merge_loras.merge([(a, 1.0), (b, 1.0)], torch.float32, verify=False)
    meta = merge_loras.merged_metadata(loras, info, torch.float32)
    assert meta["ss_base_model_version"] == "base1"
    assert "ss_network_module" not in meta  # inputs disagree
    assert meta["ss_network_dim"] == meta["ss_network_alpha"] == "3"
