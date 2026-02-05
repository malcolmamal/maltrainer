"""
Configuration loading and validation utilities.
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def save_config_to_output(config: Dict[str, Any], output_path: str):
    """Save config to output directory for reproducibility."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config_save_path = output_path / "config.yaml"
    
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
