"""The command line every LoRA trainer takes, and the start of a run from it."""

from __future__ import annotations

import argparse
from typing import Any

from maltrainer.utils.config import save_config_to_output
from maltrainer.utils.logging_utils import setup_logging


def training_arguments(description: str) -> argparse.ArgumentParser:
    """--config, --dry-run and --dry-run-step-time; a trainer may add its own."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Simulate training without loading models")
    parser.add_argument("--dry-run-step-time", type=float, default=0.025, help="Seconds per simulated step")
    return parser


def start_run(config: dict[str, Any]) -> None:
    """Logging as the config asks, and a copy of the config in the output folder."""
    setup_logging(config.get("logging", {}))
    if config.get("output"):
        save_config_to_output(config, config["output"]["path"])
