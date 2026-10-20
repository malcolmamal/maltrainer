"""scripts/train_from_template.py: fill a template for a dataset, write the config, hand it to the trainer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

import train_from_template as tft
from tests.scripts.template_world import ROOT, Launch, install_launch, make_world, rendered_config, run


@pytest.fixture
def launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Launch:
    return install_launch(monkeypatch, tmp_path)


@pytest.mark.parametrize("model", sorted(tft.MODELS))
def test_every_model_entry_points_at_a_trainer_templates_and_model_keys_that_exist(model: str) -> None:
    script, template, by_optimizer, files = tft.MODELS[model]

    assert (ROOT / script).is_file()
    for path in [template, *by_optimizer.values()]:
        config = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        assert set(files) <= set(config["model"]), path
        assert all("[comfyui_models]" in config["model"][key] for key in files)


@pytest.mark.usefixtures("launch")
def test_the_filled_config_names_the_dataset_and_the_model_files_found_anywhere_under_the_models_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    datasets, models = make_world(tmp_path)

    code = run(monkeypatch, "krea2", "alice", "--datasets", datasets, "--comfyui-models", models)

    config = rendered_config(tmp_path)
    assert code == 0
    assert config["dataset"]["path"] == str(datasets / "alice")
    assert (config["dataset"]["trigger"], config["dataset"]["default_caption"]) == ("alice", "a photo of alice")
    assert config["model"]["transformer_path"] == str(models / "diffusion_models" / "krea2_raw_fp8_scaled.safetensors")
    assert config["model"]["vae_path"] == str(models / "vae" / "nested" / "qwen_image_vae.safetensors")
    assert config["output"]["path"] == "./output/krea2/alice"
    assert "Config written to " in capsys.readouterr().out
    assert "[" not in (tmp_path / "output" / "krea2" / "alice" / "config.yaml").read_text(encoding="utf-8")


def test_the_trainer_runs_with_the_config_from_the_repo_root_and_its_exit_code_is_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch: Launch
) -> None:
    datasets, models = make_world(tmp_path, "moss_image")
    launch.code = 7

    code = run(monkeypatch, "moss_image", "alice", "--datasets", datasets, "--comfyui-models", models)

    assert code == 7
    assert launch.commands == [
        (
            [
                sys.executable,
                str(ROOT / "train_moss_image_lora.py"),
                "--config",
                str(Path("output/moss_image/alice/config.yaml")),
            ],
            ROOT,
        )
    ]


@pytest.mark.parametrize(
    ("model", "flags", "optimizer"),
    [
        ("krea2", ["--optimizer", "prodigy"], "prodigy"),
        ("moss_image", ["--optimizer", "adamw8bit"], "adamw8bit"),
        ("moss_image", [], "automagic3"),
        ("flux3_klein", ["--optimizer", "prodigy"], "prodigy"),
    ],
)
@pytest.mark.usefixtures("launch")
def test_the_optimizer_picks_its_template_when_the_model_has_one_and_is_set_in_the_config_always(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, flags: list[str], optimizer: str
) -> None:
    datasets, models = make_world(tmp_path, model)

    run(monkeypatch, model, "alice", "--datasets", datasets, "--comfyui-models", models, *flags)

    chosen = rendered_config(tmp_path, model)["train"]["optimizer"]
    assert chosen == optimizer


@pytest.mark.usefixtures("launch")
def test_learning_rate_steps_epochs_and_trigger_override_the_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets, models = make_world(tmp_path)
    flags = ["--learning-rate", "0.002", "--epochs", "5", "--trigger", "zed"]

    run(monkeypatch, "krea2", "alice", "--datasets", datasets, "--comfyui-models", models, *flags)

    config = rendered_config(tmp_path)
    assert (config["train"]["learning_rate"], config["train"]["num_epochs"], config["train"]["max_steps"]) == (
        0.002,
        5,
        0,
    )
    assert (config["dataset"]["trigger"], config["dataset"]["default_caption"]) == ("zed", "a photo of zed")


def test_max_steps_wins_over_epochs_and_the_dry_run_flag_reaches_the_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch: Launch
) -> None:
    datasets, models = make_world(tmp_path)

    run(
        monkeypatch,
        "krea2",
        "alice",
        "--datasets",
        datasets,
        "--comfyui-models",
        models,
        "--epochs",
        "5",
        "--max-steps",
        "300",
        "--dry-run",
    )

    assert rendered_config(tmp_path)["train"]["max_steps"] == 300
    assert launch.commands[0][0][-1] == "--dry-run"
