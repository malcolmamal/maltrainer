"""scripts/boost_text_fusion.py on the command line: names, factors, overwrite rules and failures."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import boost_text_fusion

FUSION = "diffusion_model.txtfusion.refiner_blocks.0.attn.wq"


def _lora(path: Path, name: str = FUSION) -> Path:
    state = {f"{name}.lora_A.weight": torch.ones(2, 4), f"{name}.lora_B.weight": torch.ones(3, 2)}
    save_file(state, str(path))
    return path


def _run(monkeypatch: pytest.MonkeyPatch, *argv: object) -> int:
    monkeypatch.setattr("sys.argv", ["boost_text_fusion.py", *map(str, argv)])
    return boost_text_fusion.main()


def _factor(path: Path) -> float:
    return float(load_file(str(path))[f"{FUSION}.lora_B.weight"][0, 0])


@pytest.mark.parametrize(
    ("flags", "name", "factor"),
    [
        ([], "a_txt3.safetensors", 3.0),
        (["--factor", "2"], "a_txt2.safetensors", 2.0),
        (["--factor", "1.5"], "a_txt1.5.safetensors", 1.5),
        (["--suffix", "_boost"], "a_boost.safetensors", 3.0),
    ],
)
def test_the_output_sits_next_to_the_input_named_after_the_factor_or_the_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], name: str, factor: float
) -> None:
    source = _lora(tmp_path / "a.safetensors")
    target = source.with_name(name)

    assert _run(monkeypatch, source, *flags) == 0

    assert _factor(target) == factor
    assert _factor(source) == 1.0


def test_an_empty_suffix_would_overwrite_the_input_and_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _lora(tmp_path / "a.safetensors")

    with pytest.raises(SystemExit):
        _run(monkeypatch, source, "--suffix", "")

    assert "the output would overwrite the input" in capsys.readouterr().err
    assert _factor(source) == 1.0


def test_several_inputs_each_get_their_own_output_and_an_output_flag_names_a_single_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    a, b = _lora(tmp_path / "a.safetensors"), _lora(tmp_path / "b.safetensors")

    assert _run(monkeypatch, a, b, "--factor", "2") == 0
    assert _run(monkeypatch, a, "--output", tmp_path / "named.safetensors") == 0

    assert (_factor(tmp_path / "a_txt2.safetensors"), _factor(tmp_path / "b_txt2.safetensors")) == (2.0, 2.0)
    assert _factor(tmp_path / "named.safetensors") == 3.0


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--factor", "0"], "--factor must be positive"),
        (["--factor", "-1"], "--factor must be positive"),
        (["--output", "x.safetensors", "--extra"], "--output works with a single input"),
    ],
)
def test_bad_arguments_are_refused_before_anything_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flags: list[str], message: str
) -> None:
    a, b = _lora(tmp_path / "a.safetensors"), _lora(tmp_path / "b.safetensors")
    inputs = [a, b] if "--extra" in flags else [a]
    flags = [f for f in flags if f != "--extra"]

    with pytest.raises(SystemExit) as stop:
        _run(monkeypatch, *inputs, *flags)

    assert stop.value.code == 2
    assert message in capsys.readouterr().err


def test_a_missing_input_an_existing_output_and_an_output_equal_to_the_input_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _lora(tmp_path / "a.safetensors")
    _lora(tmp_path / "a_txt3.safetensors")
    problems: list[tuple[list[object], str]] = [
        ([tmp_path / "nope.safetensors"], "not found: "),
        ([a], "a_txt3.safetensors exists; pass --force to overwrite"),
        ([a, "--output", a], "the output would overwrite the input"),
    ]

    for argv, message in problems:
        with pytest.raises(SystemExit) as stop:
            _run(monkeypatch, *argv)
        assert stop.value.code == 2
        assert message in capsys.readouterr().err


def test_force_overwrites_an_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = _lora(tmp_path / "a.safetensors")
    stale = _lora(tmp_path / "a_txt3.safetensors")
    save_file({"stale": torch.zeros(1)}, str(stale))

    assert _run(monkeypatch, a, "--force") == 0

    assert _factor(stale) == 3.0


def test_an_input_without_text_fusion_adapters_fails_the_run_but_not_the_other_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plain = _lora(tmp_path / "plain.safetensors", "diffusion_model.blocks.0.attn.wq")
    good = _lora(tmp_path / "good.safetensors")

    code = _run(monkeypatch, plain, good)

    captured = capsys.readouterr()
    assert code == 1
    assert "  error: plain.safetensors: no text-fusion block adapters" in captured.err
    assert not (tmp_path / "plain_txt3.safetensors").exists()
    assert _factor(tmp_path / "good_txt3.safetensors") == 3.0
    assert captured.out.splitlines()[0] == str(plain)
