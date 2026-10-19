"""
Logging utilities.
"""

import logging
import sys
from typing import Any, Dict

LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def log_level(level: Any) -> int:
    """``logging.level`` from a config: a name in any case ("info", "DEBUG") or a number."""
    if isinstance(level, int) and not isinstance(level, bool):
        return level
    name = str(level).strip().upper()
    if name not in LEVELS:
        raise ValueError(f"logging.level must be one of {', '.join(LEVELS)} (any case) or a number, got {level!r}")
    return int(getattr(logging, name))


def setup_logging(logging_config: Dict[str, Any]):
    """Setup logging configuration."""
    level = log_level(logging_config.get("level", "INFO"))
    format_str = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.basicConfig(level=level, format=format_str, handlers=[logging.StreamHandler(sys.stdout)])
