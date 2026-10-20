"""What the Krea 2 finetune's parts read from the trainer (the methods stay on the trainer: tests patch them)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import torch

from maltrainer.optimizers import Automagic3

STATE_MODEL = "model.safetensors"
STATE_OPTIMIZER = "optimizer.pt"
STATE_TRAINER = "trainer_state.pt"


class FinetuneHost(Protocol):
    config: dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    run_metadata: dict[str, object]
    samples: list[tuple[str, str]]
    is_main_process: bool
    progress_logger: logging.Logger
    transformer: Any
    rotator: Any
    always_on: list[tuple[str, torch.nn.Parameter]]
    always_optimizer: Automagic3 | None
    key_prefix: str
    resume_dir: Path | None
    resume_state: dict[str, Any] | None
    global_step: int
    epoch: int
    position: int
    epoch_rng: tuple[object, ...] | None
    last_saved_path: str | None
    lora_network: Any

    def _epoch_order(self) -> list[int]: ...
    def _rotate_every(self) -> int: ...
    def _window_at(self, step: int) -> int: ...
    def _make_optimizer(self, params: list[torch.nn.Parameter], name: str) -> Automagic3: ...
    def _dataset_fingerprint(self) -> str: ...
    def _training_step(
        self, index: int, shift_logged: list[bool] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def _log_step(self, step: int, total: int, loss: torch.Tensor, epoch: int, num_epochs: int) -> None: ...
    def save_checkpoint(self, step: int | str) -> bool: ...
    def _name(self) -> str: ...
    def _model_bytes(self) -> int: ...
    def _save_state(self, state_dir: Path, weights: Path) -> None: ...
