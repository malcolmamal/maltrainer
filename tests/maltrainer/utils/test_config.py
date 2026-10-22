from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from maltrainer.utils.config import load_config, save_config_to_output


def test_a_yaml_config_loads(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("model:\n  name: moss\ntrain:\n  steps: 10\n", encoding="utf-8")

    assert load_config(str(path)) == {"model": {"name": "moss"}, "train": {"steps": 10}}


def test_a_missing_config_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("absent.yaml")):
        load_config(str(tmp_path / "absent.yaml"))


def test_the_saved_config_keeps_key_order_and_round_trips(tmp_path: Path) -> None:
    config = {"train": {"steps": 10}, "model": {"name": "moss", "dtype": "bf16"}}

    save_config_to_output(config, str(tmp_path / "run" / "out"))

    saved = tmp_path / "run" / "out" / "config.yaml"
    assert list(yaml.safe_load(saved.read_text(encoding="utf-8"))) == ["train", "model"]
    assert load_config(str(saved)) == config


def test_the_saved_config_is_block_style_and_goes_into_a_folder_that_already_exists(tmp_path: Path) -> None:
    save_config_to_output({"train": {"steps": 10}}, str(tmp_path))
    save_config_to_output({"train": {"steps": 20}}, str(tmp_path))

    expected = "train:" + chr(10) + "  steps: 20" + chr(10)
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == expected
