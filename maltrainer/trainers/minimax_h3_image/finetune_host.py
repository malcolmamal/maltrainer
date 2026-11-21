"""What the H3 Image finetune's parts read from the trainer (the methods stay on the trainer: tests patch them)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import torch

from maltrainer.finetune.rotation_int8 import Int8WindowRotator
from maltrainer.optimizers import Automagic3
from maltrainer.trainers.minimax_h3.sample import H3Sample

STATE_WEIGHTS = "weights.safetensors"  # the bf16 value of every trained weight (a resume needs more than the INT8 file)
STATE_OPTIMIZER = "optimizer.pt"
STATE_TRAINER = "trainer_state.pt"


class FinetuneHost(Protocol):
    config: dict[str, Any]
    device: torch.device
    dtype: torch.dtype
    run_metadata: dict[str, object]
    samples: list[H3Sample]
    progress_logger: logging.Logger
    is_main_process: bool
    transformer: Any
    sigmas: tuple[float, ...]  # the checkpoint's noise levels, with the final 0
    lora_network: Any  # the transformer again: the inherited step calls whatever this holds
    adapter_layers: Any  # the frozen training adapter's layers, when it is on
    rotator: Int8WindowRotator
    always_on: list[tuple[str, torch.nn.Parameter]]
    always_optimizer: Automagic3 | None
    optimizer: Any  # the active window's optimizer once one is up, for the control API
    resume_dir: Path | None
    resume_state: dict[str, Any] | None
    global_step: int
    epoch: int
    position: int
    epoch_rng: tuple[object, ...] | None
    last_saved_path: str | None
    _steps_taken: int

    def _epoch_order(self) -> list[int]: ...
    def _epoch_size(self) -> int: ...
    def _rotate_every(self) -> int: ...
    def _window_at(self, step: int) -> int: ...
    def _make_optimizer(self, params: list[torch.nn.Parameter], name: str) -> Automagic3: ...
    def _dataset_fingerprint(self) -> str: ...
    def _name(self) -> str: ...
    def _training_step(self, sample: H3Sample, flow_shift: float) -> tuple[torch.Tensor, dict[str, float]]: ...
    def save_checkpoint(self, step: int | str) -> None: ...
    def _save_state(self, state_dir: Path) -> None: ...
