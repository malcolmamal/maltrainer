"""scripts/merge_lora_into_model.py from a YAML file: every setting, the command line on top, mistakes, the sample."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from merge_into_model_plan import from_yaml
from tests.scripts.merge_into_model_world import P, model_file, near, run

ROOT = Path(__file__).resolve().parents[2]


def test_a_yaml_config_gives_every_setting_and_the_command_line_adds_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = model_file(tmp_path / "raw.safetensors", 1)
    finetune, other = near(tmp_path / "ft.safetensors", raw, 0.05, 2), near(tmp_path / "o.safetensors", raw, 0.05, 3)
    config = tmp_path / "merge.yaml"
    config.write_text(
        f"base: {raw.as_posix()}\noutput: {(tmp_path / 'yaml_out.safetensors').as_posix()}\ndtype: float16\n"
        f"models:\n  - path: {other.as_posix()}\n    weight: 0.25\n"
        f"add_difference:\n  - model: {finetune.as_posix()}\n    base: {raw.as_posix()}\n",
        encoding="utf-8",
    )

    assert run(monkeypatch, "--config", str(config)) == 0
    r, f, o = (load_file(str(p)) for p in (raw, finetune, other))
    key = f"{P}blocks.0.lin.weight"
    merged = load_file(str(tmp_path / "yaml_out.safetensors"))[key]
    assert merged.dtype == torch.float16
    torch.testing.assert_close(merged.float(), 0.75 * r[key] + 0.25 * o[key] + (f[key] - r[key]), atol=2e-3, rtol=2e-3)

    cli_out = tmp_path / "cli_out.safetensors"
    assert (
        run(
            monkeypatch,
            "--config",
            str(config),
            str(raw),
            str(cli_out),
            "--dtype",
            "float32",
            "--model",
            f"{other}:0.25",
        )
        == 0
    )
    torch.testing.assert_close(load_file(str(cli_out))[key], 0.5 * r[key] + 0.5 * o[key] + (f[key] - r[key]))


@pytest.mark.parametrize(
    "text, message",
    [
        ("base: a\nwhatever: 1\n", r"unknown keys \['whatever'\]"),
        ("models:\n  - path: a\n", r"models\[0\] needs path, weight"),
        (
            "add_difference:\n  - model: a\n    base: b\n    weight: 2\n",
            r"add_difference\[0\] has unknown keys \['weight'\]",
        ),
        ("dtype: int8\n", r"dtype must be one of"),
        ("loras: a.safetensors\n", r"'loras' must be a list"),
    ],
)
def test_a_config_with_a_mistake_is_refused_with_where_it_is(tmp_path: Path, text: str, message: str) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        from_yaml(config)


def test_the_sample_config_parses_to_the_krea2_turbo_recipe() -> None:
    plan = from_yaml(ROOT / "templates" / "examples" / "merge_into_model_krea2.yaml")
    assert plan.base is not None and plan.base.name == "krea2_turbo_fp8_scaled.safetensors"
    assert plan.dtype == "bfloat16" and plan.terms.weighted == []
    assert [(b.name, c.name, s) for b, c, s in plan.terms.differences] == [
        ("krea2_ft_krea.safetensors", "_base_dequantized_bf16.safetensors", 1.0)
    ]
    assert [(p.name, s) for p, s in plan.loras] == [("Krea2_TextFusion_Refusal_Reduction.safetensors", 1.0)]


@pytest.mark.parametrize(
    "args, message",
    [
        (["--add-diff", "only_one.safetensors"], "give MODEL,BASE or MODEL,BASE:STRENGTH"),
        ([], "nothing to merge"),
        (["--model", "missing.safetensors:0.5"], "not found: missing.safetensors"),
    ],
)
def test_command_line_mistakes_stop_before_anything_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], args: list[str], message: str
) -> None:
    a = model_file(tmp_path / "a.safetensors", 1)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        run(monkeypatch, str(a), str(tmp_path / "out.safetensors"), *args)
    assert message in capsys.readouterr().err and not (tmp_path / "out.safetensors").exists()
