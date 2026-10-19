"""
Logging utilities.
"""

import logging
import sys
from typing import Any, Dict


def setup_logging(logging_config: Dict[str, Any]):
    """Setup logging configuration."""
    level = logging_config.get("level", "INFO")
    format_str = logging_config.get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logging.basicConfig(
        level=getattr(logging, level),
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
