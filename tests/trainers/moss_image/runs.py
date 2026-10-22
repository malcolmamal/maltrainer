"""Running a trainer end to end on the tiny world, and comparing what two runs left behind."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import train_moss_image_finetune as finetune
import train_moss_image_lora as lora


def run_finetune(config: dict[str, Any]) -> finetune.MOSSImageFinetuneTrainer:
    trainer = finetune.MOSSImageFinetuneTrainer(config)
    trainer.run()
    return trainer


def run_lora(config: dict[str, Any]) -> lora.MOSSImageLoRATrainer:
    trainer = lora.MOSSImageLoRATrainer(config)
    trainer.run()
    return trainer


def optimizer_state(optimizer: torch.optim.Optimizer) -> dict[tuple[int, str], Any]:
    """Every entry of the optimizer's per-parameter state, keyed by (parameter index, name)."""
    return {
        (index, name): value
        for index, entry in optimizer.state_dict()["state"].items()
        for name, value in entry.items()
    }


def same_values(left: dict[Any, Any], right: dict[Any, Any]) -> bool:
    """Equal keys and, tensor by tensor, bit-identical values."""
    if left.keys() != right.keys():
        return False
    return all(
        torch.equal(left[key], right[key]) if torch.is_tensor(left[key]) else left[key] == right[key] for key in left
    )


def prepared_finetune(config: dict[str, Any]) -> finetune.MOSSImageFinetuneTrainer:
    """A trainer with its data cached, its model loaded and its optimizer built, before any step."""
    trainer = finetune.MOSSImageFinetuneTrainer(config)
    trainer.precache_dataset()
    trainer.load_transformer_and_lora()
    trainer.setup_optimizer()
    return trainer


def changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> set[str]:
    """The names whose tensors differ."""
    return {name for name in before if not torch.equal(before[name], after[name])}


def base_prediction(
    base: torch.nn.Module, x0: torch.Tensor, sigma: torch.Tensor, seed: int | torch.Tensor, context: torch.Tensor
) -> torch.Tensor:
    """What the unchanged model answers for one recorded prior input (x0 [C, h, w], context [L, D])."""
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(int(seed)), dtype=torch.float32)
    s = sigma.reshape(1, 1, 1)
    noisy = (1.0 - s) * x0.float() + s * noise
    with torch.no_grad():
        return base(noisy.unsqueeze(0).to(torch.bfloat16), sigma.reshape(1), context.unsqueeze(0).to(torch.bfloat16))[0]


def cached_finetune(config: dict[str, Any]) -> finetune.MOSSImageFinetuneTrainer:
    """A trainer that has scanned and cached its dataset (nothing else)."""
    trainer = finetune.MOSSImageFinetuneTrainer(config)
    trainer.precache_dataset()
    return trainer


def record_sample_order(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make every finetune training step note which sample it trained on; the list fills as the run goes."""
    seen: list[int] = []
    original = finetune.MOSSImageFinetuneTrainer._training_step

    def noting(self: finetune.MOSSImageFinetuneTrainer, index: int) -> torch.Tensor:
        seen.append(index)
        return original(self, index)

    monkeypatch.setattr(finetune.MOSSImageFinetuneTrainer, "_training_step", noting)
    return seen


def prepared_lora(config: dict[str, Any]) -> lora.MOSSImageLoRATrainer:
    """A LoRA trainer with its data cached, its model and adapters loaded and its optimizer built."""
    trainer = lora.MOSSImageLoRATrainer(config)
    trainer.precache_dataset()
    trainer.load_transformer_and_lora()
    trainer.setup_optimizer()
    return trainer


def cached_lora(config: dict[str, Any]) -> lora.MOSSImageLoRATrainer:
    """A LoRA trainer that has scanned and cached its dataset (nothing else)."""
    trainer = lora.MOSSImageLoRATrainer(config)
    trainer.precache_dataset()
    return trainer


def progress_lines(ws: Any) -> list[str]:
    """The ``Step:`` lines of the run's training.log (the progress logger writes only there)."""
    return [line for line in (ws.out / "training.log").read_text(encoding="utf-8").splitlines() if "Step:" in line]
