"""Device reporting and the non-main process: what the trainers log, print and save depending on who they are."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from accelerate import Accelerator

import train_zimage_embedding as embedding
import train_zimage_lora as lora

from .world import Workspace

KINDS = pytest.mark.parametrize("kind", ["lora", "embedding"])


def _module(kind: str) -> Any:
    return lora if kind == "lora" else embedding


def _device_logger(kind: str) -> str:
    """Where the trainer's device lines come from (the LoRA trainer reports through the shared session)."""
    return "maltrainer.training.session" if kind == "lora" else embedding.__name__


def _trainer(ws: Workspace, kind: str) -> Any:
    if kind == "lora":
        return lora.ZImageLoRATrainer(ws.lora_config())
    return embedding.ZImageEmbeddingTrainer(ws.embedding_config())


@KINDS
def test_a_trainer_on_a_cuda_machine_logs_the_gpu_name_and_its_memory(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "Tiny GPU")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=2 * 1024**3))

    with caplog.at_level(logging.INFO, logger=_device_logger(kind)):
        trainer = _trainer(workspace, kind)

    assert trainer.device == torch.device("cuda")
    assert caplog.messages[:2] == ["Training on device: cuda (Tiny GPU)", "Total VRAM: 2.00 GB"]


@KINDS
def test_a_trainer_on_a_cpu_logs_only_the_device(
    workspace: Workspace, caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    with caplog.at_level(logging.INFO, logger=_device_logger(kind)):
        trainer = _trainer(workspace, kind)

    assert trainer.device == torch.device("cpu") and trainer.dtype == torch.float32
    assert caplog.messages == ["Training on device: cpu (CPU)"]


class Worker(Accelerator):
    """An accelerator that is one of several processes and not the main one."""

    @property
    def is_main_process(self) -> bool:
        return False


@KINDS
def test_a_process_that_is_not_the_main_one_logs_no_device_prints_no_banner_and_saves_nothing(
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    workspace.add_images(2)
    monkeypatch.setattr(_module(kind), "Accelerator", Worker)
    with caplog.at_level(logging.INFO, logger=_device_logger(kind)):
        trainer = _trainer(workspace, kind)
    assert trainer.is_main_process is False and "Training on device" not in caplog.text

    trainer.run()

    assert "--- Training Configuration ---" not in capsys.readouterr().out
    assert list(workspace.out.glob("*.safetensors")) == []
