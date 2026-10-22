"""A LoRA trainer watching its config file, for the tests of live sample: reloads."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import yaml

import train_moss_image_lora as lora

from .world import SAMPLE, Workspace

NEW = "a portrait of someone new"


def watched(ws: Workspace, **sections: Any) -> tuple[lora.MOSSImageLoRATrainer, Callable[[dict[str, Any]], None]]:
    """A trainer watching ``run.yaml`` and a function that rewrites that file (always with a newer mtime)."""
    path, config = ws.root / "run.yaml", ws.lora_config(sample=SAMPLE, **sections)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    trainer = lora.MOSSImageLoRATrainer(config)
    trainer.config_path = path
    trainer._watch_config()
    version = [path.stat().st_mtime_ns]

    def rewrite(changed: dict[str, Any]) -> None:
        path.write_text(yaml.safe_dump({**config, **changed}), encoding="utf-8")
        version[0] += 10_000_000_000
        os.utime(path, ns=(version[0], version[0]))

    return trainer, rewrite


def wait_for_encoding(trainer: lora.MOSSImageLoRATrainer) -> None:
    assert trainer._encoding is not None
    trainer._encoding.join(timeout=10)
