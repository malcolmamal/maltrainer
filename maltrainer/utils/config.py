"""
Configuration loading and validation utilities.
"""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as error:
            raise ValueError(f"{config_path} is not valid YAML: {error}") from error


def save_config_to_output(config: dict[str, Any], output_path: str):
    """Save config to output directory for reproducibility."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    config_save_path = output_path / "config.yaml"

    with config_save_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
