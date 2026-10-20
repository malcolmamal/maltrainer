"""The Z-Image trainers' run order: seed, then the dry run or each stage in turn, then the summary line; an
interrupt ends quietly, any other error is logged with its traceback and raised."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import torch
from accelerate.utils import set_seed

logger = logging.getLogger(__name__)


class Pipeline:
    config: dict[str, Any]
    dry_run: bool
    last_saved_path: str | None

    def dry_run_training(self) -> None:
        raise NotImplementedError

    def _stages(self) -> list[Callable[[], None]]:
        """The methods that load, build and train, in order (looked up when the run starts)."""
        raise NotImplementedError

    def _log_summary(self, start_time: float) -> None:
        elapsed = max(time.time() - start_time, 0.0)
        output_path = self.last_saved_path or self.config.get("output", {}).get("path", "")
        logger.info("Summary: ran for %.2fs, last output saved to %s", elapsed, output_path)

    def run(self) -> None:
        """Run the complete training pipeline."""
        try:
            set_seed(self.config.get("seed", 42))
            start_time = time.time()
            if self.dry_run:
                self.dry_run_training()
                self._log_summary(start_time)
                return
            for stage in self._stages():
                stage()
            self._log_summary(start_time)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user. Exiting...")
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            raise e
