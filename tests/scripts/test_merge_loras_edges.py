"""merge_loras.py: files it refuses to read, a config with no output, and the warnings and notes it prints."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

import merge_loras
from tests.scripts.helpers import write_lora
from tests.scripts.test_merge_loras import D1, D2, U1, U2, run


def test_a_down_weight_without_its_up_weight_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "half.safetensors"
    save_file({"m.lora_A.weight": D1}, str(path))

    with pytest.raises(ValueError, match=r"m\.lora_A\.weight has no matching m\.lora_B\.weight"):
        merge_loras.read_lora(path)


def test_a_lora_that_is_not_a_plain_linear_one_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "conv.safetensors"
    save_file({"m.lora_A.weight": torch.ones(2, 4, 3, 3), "m.lora_B.weight": torch.ones(3, 2, 1, 1)}, str(path))

    with pytest.raises(ValueError, match="m is not a plain linear LoRA"):
        merge_loras.read_lora(path)


def test_a_config_without_an_output_needs_one_on_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, None)})
    config = tmp_path / "c.yaml"
    config.write_text(yaml.dump({"loras": [{"path": str(a)}]}))

    with pytest.raises(SystemExit) as raised:
        run(monkeypatch, "--config", config)

    assert raised.value.code == 2
    assert "no output" in capsys.readouterr().err


def test_inputs_from_different_base_models_are_merged_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, None)}, metadata={"ss_base_model_version": "one"})
    b = write_lora(tmp_path / "b.safetensors", {"m": (D2, U2, None)}, metadata={"ss_base_model_version": "two"})

    assert run(monkeypatch, tmp_path / "out.safetensors", a, b) == 0

    assert "different base models: ['one', 'two']" in capsys.readouterr().err


def test_modules_present_in_only_some_inputs_are_counted_in_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = write_lora(tmp_path / "a.safetensors", {"m": (D1, U1, None), "n": (D2, U2, None)})
    b = write_lora(tmp_path / "b.safetensors", {"m": (D2, U2, None)})

    assert run(monkeypatch, tmp_path / "out.safetensors", a, b, "--verify") == 0

    out = capsys.readouterr().out
    assert "1 modules came from only some of the inputs" in out
    assert "verify: worst module differs" in out
