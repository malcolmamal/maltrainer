from __future__ import annotations

import json
import sys

import pytest
import torch
import yaml
from safetensors.torch import save_file

import inspect_lora_metadata as inspect_meta

CONFIG = {"model": {"type": "zimage"}, "train": {"steps": 5}}


def make(path, metadata):
    save_file({"w": torch.zeros(1)}, str(path), metadata=metadata)
    return path


def run(monkeypatch, *argv) -> int:
    monkeypatch.setattr("sys.argv", ["inspect_lora_metadata.py", *map(str, argv)])
    return inspect_meta.main()


@pytest.fixture
def lora(tmp_path):
    meta = {"ss_network_dim": "16", "ss_optimizer": "adamw", "maltrainer_config": json.dumps(CONFIG), "step": "250"}
    return make(tmp_path / "a.safetensors", meta)


def test_summary_lists_recognised_keys(lora, monkeypatch, capsys):
    assert run(monkeypatch, lora) == 0
    out = capsys.readouterr().out
    assert "rank  " in out and "16" in out and "adamw" in out
    assert "embedded maltrainer config: yes" in out
    assert "chars, use --all" not in out  # short config blob is not elided


def test_summary_without_metadata(tmp_path, monkeypatch, capsys):
    assert run(monkeypatch, make(tmp_path / "n.safetensors", None)) == 0
    assert "no metadata at all" in capsys.readouterr().out


def test_all_and_json_outputs(lora, monkeypatch, capsys):
    run(monkeypatch, lora, "--all")
    lines = capsys.readouterr().out.splitlines()
    assert lines == sorted(lines) and "ss_network_dim=16" in lines
    run(monkeypatch, lora, "--json")
    assert json.loads(capsys.readouterr().out)["ss_optimizer"] == "adamw"


def test_config_recovery(lora, tmp_path, monkeypatch, capsys):
    target = tmp_path / "r.yaml"
    assert run(monkeypatch, lora, "--config", "-o", target) == 0
    assert yaml.safe_load(target.read_text(encoding="utf-8")) == CONFIG
    assert run(monkeypatch, lora, "--config") == 0
    assert yaml.safe_load(capsys.readouterr().out.replace(f"Wrote {target}\n", "")) == CONFIG


def test_config_missing_returns_1(tmp_path, monkeypatch, capsys):
    assert run(monkeypatch, make(tmp_path / "n.safetensors", {"a": "b"}), "--config") == 1
    assert "No embedded config" in capsys.readouterr().err


def test_diff_ignores_step_and_config_blob(lora, tmp_path, monkeypatch, capsys):
    other = make(tmp_path / "b.safetensors", {"ss_network_dim": "32", "ss_optimizer": "adamw", "step": "9"})
    assert run(monkeypatch, lora, "--diff", other) == 0
    out = capsys.readouterr().out
    assert "1 difference(s)" in out and "ss_network_dim" in out
    assert "ss_optimizer" not in out.split("\n\n", 1)[1]
    assert run(monkeypatch, lora, "--diff", lora) == 0
    assert "Identical" in capsys.readouterr().out


def test_missing_and_malformed_files_return_1(tmp_path, lora, monkeypatch, capsys):
    assert run(monkeypatch, tmp_path / "nope.safetensors") == 1
    assert run(monkeypatch, lora, "--diff", tmp_path / "nope.safetensors") == 1
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a safetensors file")
    assert run(monkeypatch, junk) == 1
    assert capsys.readouterr().err.count("junk.safetensors") == 1


def test_fmt_pretty_prints_json_and_big_numbers():
    assert inspect_meta._fmt('{"b": 1, "a": 2}') == '{"a": 2, "b": 1}'
    assert inspect_meta._fmt("{broken") == "{broken"
    assert inspect_meta._fmt("1234567") == "1,234,567"
    assert inspect_meta._fmt("123456") == "123456"
    assert inspect_meta._abbreviate("x" * 200).endswith("... (200 chars)")


def test_a_file_from_another_tool_lists_its_keys_and_says_it_is_not_ours(tmp_path, monkeypatch, capsys):
    bulky = "x" * 200
    path = make(tmp_path / "k.safetensors", {"ss_datasets": bulky, "ss_tag_frequency": "{}", "modelspec.title": "t"})

    assert run(monkeypatch, path) == 0

    out = capsys.readouterr().out
    assert "recognised training settings" not in out
    assert "ss_datasets" in out and "<200 chars, use --all or --json>" in out
    assert "ss_tag_frequency" in out and bulky not in out
    assert "probably not a maltrainer checkpoint" in out
    assert "embedded maltrainer config: no" in out


def test_the_config_comes_out_as_json_when_pyyaml_is_missing(lora, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "yaml", None)

    assert run(monkeypatch, lora, "--config") == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == CONFIG
    assert "PyYAML unavailable" in captured.err
