from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import files, registry

Installer = Callable[..., list[Path]]
QWEN_FILES = [
    "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
    "vae/qwen_image_2.1_vae_bf16.safetensors",
    "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
]


def _models(client: TestClient) -> dict[str, dict]:
    return {item["id"]: item for item in client.get("/api/models").json()}


def test_every_registered_model_is_listed_in_registry_order(client: TestClient) -> None:
    assert list(_models(client)) == list(registry.MODELS)


@pytest.mark.parametrize("model_id", list(registry.MODELS))
def test_each_model_offers_its_optimizers_with_their_default_rates(client: TestClient, model_id: str) -> None:
    spec = registry.MODELS[model_id]

    listed = _models(client)[model_id]

    assert listed["default_optimizer"] == spec.default_optimizer
    assert [(o["name"], o["default_lr"]) for o in listed["optimizers"]] == [
        (o.name, o.default_lr) for o in spec.optimizers
    ]


def test_krea2_offers_automagic3_prodigy_and_adamw8bit(client: TestClient) -> None:
    assert [o["name"] for o in _models(client)["krea2"]["optimizers"]] == ["automagic3", "prodigy", "adamw8bit"]


def test_the_listing_carries_the_duration_defaults_and_auto_step_range(client: TestClient) -> None:
    models = _models(client)

    assert {key: models["zimage"][key] for key in ("default_duration_mode", "default_epochs", "auto_steps")} == {
        "default_duration_mode": "epochs",
        "default_epochs": 100,
        "auto_steps": False,
    }
    qwen = models["qwen_image_2_1"]
    assert (qwen["auto_steps"], qwen["target_epochs"], qwen["min_auto_steps"], qwen["max_auto_steps"]) == (
        True,
        70,
        1000,
        2500,
    )
    assert qwen["sec_per_step"] == 0.62
    assert qwen["resolution_note"].startswith("Training resolution is square")


def test_a_model_without_checkpoint_requirements_is_always_available(client: TestClient) -> None:
    zimage = _models(client)["zimage"]

    assert (zimage["models_root"], zimage["missing_files"]) == (None, [])


def test_an_empty_models_folder_reports_every_required_file_missing(client: TestClient, workspace: Path) -> None:
    qwen = _models(client)["qwen_image_2_1"]

    assert qwen["models_root"] == str((workspace / "models").resolve())
    assert qwen["missing_files"] == QWEN_FILES


def test_an_unset_models_folder_reports_every_file_missing_with_no_root(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(files.COMFYUI_MODELS_ENV)

    qwen = _models(client)["qwen_image_2_1"]

    assert (qwen["models_root"], qwen["missing_files"]) == (None, QWEN_FILES)


def test_installed_checkpoints_are_found_and_the_rest_stay_missing(
    client: TestClient, install_models: Installer
) -> None:
    install_models("qwen_image_2_1")[1].unlink()

    assert _models(client)["qwen_image_2_1"]["missing_files"] == [QWEN_FILES[1]]


def test_a_checkpoint_is_found_anywhere_below_its_subfolder(client: TestClient, install_models: Installer) -> None:
    install_models("flux3_klein", nested="qwen/v2")

    assert _models(client)["flux3_klein"]["missing_files"] == []


def test_fibo2_is_offered_and_its_checkpoints_are_found_below_the_fibo2_folder(
    client: TestClient, install_models: Installer
) -> None:
    install_models("fibo2", nested="Fibo2")

    entry = _models(client)["fibo2"]
    assert entry["missing_files"] == []
    assert entry["label"] == "Fibo 2.0 (Dev)"


def test_resolved_files_map_config_keys_to_absolute_paths(install_models: Installer) -> None:
    paths = install_models("krea2", nested="deep")

    found, missing = files.resolve_model_files(registry.MODELS["krea2"])

    assert missing == []
    assert found == dict(zip(("transformer_path", "vae_path", "text_encoder_path"), map(str, paths), strict=True))


def test_a_file_name_that_is_a_folder_does_not_count_as_a_checkpoint(workspace: Path) -> None:
    (workspace / "models" / "vae" / "flux2_vae.safetensors").mkdir(parents=True)

    _, missing = files.resolve_model_files(registry.MODELS["flux3_klein"])

    assert "vae/flux2_vae.safetensors" in missing


def test_a_lookup_is_cached_until_it_is_older_than_the_cache_window(install_models: Installer) -> None:
    spec = registry.MODELS["flux3_klein"]
    assert len(files.resolve_model_files(spec)[1]) == 3
    install_models("flux3_klein")

    assert len(files.resolve_model_files(spec)[1]) == 3

    for key, (stamp, found, missing) in list(files._model_file_cache.items()):
        files._model_file_cache[key] = (stamp - files.MODEL_FILE_CACHE_SECONDS - 1, found, missing)

    assert files.resolve_model_files(spec)[1] == []


def test_callers_cannot_change_the_cached_lookup(install_models: Installer) -> None:
    spec = registry.MODELS["flux3_klein"]
    install_models("flux3_klein")
    found, missing = files.resolve_model_files(spec)
    found.clear()
    missing.append("x")

    again_found, again_missing = files.resolve_model_files(spec)

    assert (len(again_found), again_missing) == (3, [])
