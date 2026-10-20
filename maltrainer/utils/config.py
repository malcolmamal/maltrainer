"""
Configuration loading and validation utilities.
"""

from pathlib import Path
from typing import Any, cast

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            return cast(dict[str, Any], yaml.safe_load(f))
        except yaml.YAMLError as error:
            raise ValueError(f"{path} is not valid YAML: {error}") from error


def save_config_to_output(config: dict[str, Any], output_path: str) -> None:
    """Save config to output directory for reproducibility."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_save_path = output_dir / "config.yaml"

    with config_save_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
