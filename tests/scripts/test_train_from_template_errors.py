"""scripts/train_from_template.py: environment variables, missing folders and files, and ``find_model``."""

from __future__ import annotations

from pathlib import Path

import pytest

import train_from_template as tft
from tests.scripts.template_world import Launch, install_launch, make_world, rendered_config, run


@pytest.fixture
def launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Launch:
    return install_launch(monkeypatch, tmp_path)


@pytest.mark.usefixtures("launch")
def test_environment_variables_supply_the_datasets_and_models_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets, models = make_world(tmp_path, "qwen_image_2_1", "bob")
    monkeypatch.setenv("DATASETS_PATH", str(datasets))
    monkeypatch.setenv("COMFYUI_MODELS_PATH", str(models))

    assert run(monkeypatch, "qwen_image_2_1", "bob") == 0

    config = rendered_config(tmp_path, "qwen_image_2_1", "bob")
    assert config["dataset"]["path"] == str(datasets / "bob")
    assert config["model"]["vae_path"].startswith(str(models))


def test_a_missing_dataset_folder_is_reported_and_nothing_is_written_orrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch: Launch, capsys: pytest.CaptureFixture[str]
) -> None:
    datasets, models = make_world(tmp_path)

    code = run(monkeypatch, "krea2", "nobody", "--datasets", datasets, "--comfyui-models", models)

    assert code == 1
    assert f"Dataset folder not found: {datasets / 'nobody'}" in capsys.readouterr().err
    assert not (tmp_path / "output").exists() and launch.commands == []


def test_model_files_that_are_missing_stop_the_run_unless_it_is_a_dryrun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, launch: Launch, capsys: pytest.CaptureFixture[str]
) -> None:
    datasets, models = make_world(tmp_path)
    (models / "vae" / "nested" / "qwen_image_vae.safetensors").unlink()
    argv = ["krea2", "alice", "--datasets", datasets, "--comfyui-models", models]

    assert run(monkeypatch, *argv) == 1
    assert f"Not found under {models}: vae/qwen_image_vae.safetensors" in capsys.readouterr().err
    assert launch.commands == []
    assert run(monkeypatch, *argv, "--dry-run") == 0

    model = rendered_config(tmp_path)["model"]
    assert model["vae_path"] == f"{models.as_posix()}/vae/qwen_image_vae.safetensors"
    assert model["transformer_path"] == str(models / "diffusion_models" / "krea2_raw_fp8_scaled.safetensors")


@pytest.mark.usefixtures("launch")
def test_without_a_models_folder_a_real_run_asks_for_one_and_a_dry_run_keeps_the_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    datasets, _ = make_world(tmp_path)

    assert run(monkeypatch, "krea2", "alice", "--datasets", datasets) == 1
    assert "Set COMFYUI_MODELS_PATH (or --comfyui-models)" in capsys.readouterr().err
    assert run(monkeypatch, "krea2", "alice", "--datasets", datasets, "--dry-run") == 0

    assert rendered_config(tmp_path)["model"]["vae_path"] == "[comfyui_models]/vae/qwen_image_vae.safetensors"


def test_find_model_prefers_the_file_directly_in_the_folder_then_searches_below(tmp_path: Path) -> None:
    (tmp_path / "vae" / "a" / "b").mkdir(parents=True)
    (tmp_path / "vae" / "a" / "b" / "deep.safetensors").touch()
    (tmp_path / "vae" / "top.safetensors").touch()
    (tmp_path / "vae" / "a" / "top.safetensors").touch()

    assert tft.find_model(tmp_path, "vae", "top.safetensors") == tmp_path / "vae" / "top.safetensors"
    assert tft.find_model(tmp_path, "vae", "deep.safetensors") == tmp_path / "vae" / "a" / "b" / "deep.safetensors"
    assert tft.find_model(tmp_path, "vae", "absent.safetensors") is None
    assert tft.find_model(tmp_path, "no_such_folder", "top.safetensors") is None
