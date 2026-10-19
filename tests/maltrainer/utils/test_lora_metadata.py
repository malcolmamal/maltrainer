from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.utils.lora_metadata import CONFIG_KEY, build_metadata, extract_config, read_metadata

CONFIG = {
    "model_type": "krea2",
    "seed": 7,
    "mixed_precision": "bf16",
    "model": {"transformer_path": "/models/krea2/transformer.safetensors", "vae_path": "C:/models/vae.safetensors"},
    "lora": {"rank": 16, "alpha": 8.0, "target_modules": ["to_q", "to_k"], "dtype": "bfloat16"},
    "train": {"optimizer": "adamw", "learning_rate": 1e-4, "gradient_accumulation_steps": 2},
    "dataset": {"resolution": [1024, 768]},
}


def test_an_empty_config_records_the_defaults_as_strings() -> None:
    metadata = build_metadata({}, step=5, embed_config=False)

    assert metadata == {
        "maltrainer_metadata_version": "1",
        "maltrainer_architecture": "minimax_h3",
        "maltrainer_trainer": "train_minimax_h3_lora.py",
        "ss_base_model_version": "minimax_h3_fl2va",
        "ss_network_module": "maltrainer.models.minimax_h3.lora",
        "ss_network_dim": "",
        "ss_network_alpha": "",
        "ss_network_args": '{"dtype": "float32", "prefix": null, "target_modules": null}',
        "ss_optimizer": "",
        "ss_learning_rate": "",
        "ss_weight_decay": "",
        "ss_lr_warmup_steps": "0",
        "ss_num_epochs": "",
        "ss_batch_size": "",
        "ss_gradient_accumulation_steps": "1",
        "ss_max_grad_norm": "",
        "ss_mixed_precision": "",
        "ss_seed": "",
        "ss_resolution": "",
        "ss_timestep_sampling": "",
        "ss_sigmoid_scale": "",
        "ss_discrete_flow_shift": "",
        "ss_transformer": "",
        "ss_vae": "",
        "ss_text_encoder": "",
        "step": "5",
        "rank": "",
        "alpha": "",
        "architecture": "minimax_h3",
    }


def test_the_architecture_picks_the_trainer_and_the_kohya_keys() -> None:
    metadata = build_metadata(CONFIG, step=100)

    assert metadata["maltrainer_trainer"] == "train_krea2_lora.py"
    assert metadata["ss_base_model_version"] == "krea2_raw"
    assert metadata["ss_network_module"] == "maltrainer.models.krea2.lora"
    assert (metadata["ss_network_dim"], metadata["rank"], metadata["alpha"]) == ("16", "16", "8.0")
    assert metadata["ss_network_args"] == '{"dtype": "bfloat16", "prefix": null, "target_modules": ["to_q", "to_k"]}'
    assert (metadata["ss_learning_rate"], metadata["ss_gradient_accumulation_steps"]) == ("0.0001", "2")
    assert (metadata["ss_seed"], metadata["ss_mixed_precision"], metadata["ss_resolution"]) == (
        "7",
        "bf16",
        "[1024, 768]",
    )


def test_base_models_are_named_by_file_name_only() -> None:
    metadata = build_metadata(CONFIG, step=1)

    assert (metadata["ss_transformer"], metadata["ss_vae"], metadata["ss_text_encoder"]) == (
        "transformer.safetensors",
        "vae.safetensors",
        "",
    )


def test_runtime_values_are_prefixed_and_win_the_architecture() -> None:
    runtime = {"model_type": "moss_image_finetune", "sigma_shift": 3.16, "compiled": True, "skipped": None}

    metadata = build_metadata(CONFIG, step=1, runtime=runtime)

    assert metadata["architecture"] == "moss_image_finetune"
    assert metadata["maltrainer_trainer"] == "train_moss_image_finetune.py"
    assert metadata["maltrainer_sigma_shift"] == "3.16"
    assert metadata["maltrainer_compiled"] == "true"
    assert metadata["maltrainer_skipped"] == ""


def test_an_unknown_architecture_keeps_its_name_with_the_default_trainer() -> None:
    metadata = build_metadata({"model_type": "future_model"}, step=1)

    assert metadata["architecture"] == "future_model"
    assert metadata["maltrainer_trainer"] == "train_minimax_h3_lora.py"


def test_the_embedded_config_keeps_key_order_and_stringifies_paths() -> None:
    config = {"train": {"steps": 3}, "model": {"transformer_path": Path("models") / "t.safetensors"}}

    blob = build_metadata(config, step=1)[CONFIG_KEY]

    assert list(json.loads(blob)) == ["train", "model"]
    assert extract_config({CONFIG_KEY: blob}) == {
        "train": {"steps": 3},
        "model": {"transformer_path": str(Path("models") / "t.safetensors")},
    }


def test_the_config_round_trips_through_a_safetensors_file(tmp_path: Path) -> None:
    path = tmp_path / "lora.safetensors"
    save_file({"w": torch.zeros(2)}, str(path), metadata=build_metadata(CONFIG, step=100))

    metadata = read_metadata(path)

    assert metadata["step"] == "100"
    assert extract_config(metadata) == CONFIG


def test_a_file_without_metadata_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "bare.safetensors"
    save_file({"w": torch.zeros(1)}, str(path))

    assert read_metadata(path) == {}
    assert extract_config({}) is None


def test_an_invalid_embedded_config_is_logged_and_ignored(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="maltrainer.utils.lora_metadata"):
        config = extract_config({CONFIG_KEY: "{not json"})

    assert config is None
    assert caplog.messages[0].startswith("Embedded config is not valid JSON: ")
