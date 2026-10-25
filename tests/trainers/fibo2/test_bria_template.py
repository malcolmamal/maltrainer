"""templates/fibo2_bria_template.yaml trains with the defaults of BRIA's fine_tuning/train_lora.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from safetensors.torch import load_file
from torch import nn

from maltrainer.training.sigma import clamped
from train_fibo2_lora import Fibo2LoRATrainer

from .runs import run_lora
from .world import Workspace

TEMPLATE = Path(__file__).resolve().parents[3] / "templates" / "fibo2_bria_template.yaml"


def bria_sections() -> dict[str, Any]:
    """The template's settings, without the paths the workspace points at its own files."""
    config: dict[str, Any] = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8").replace("[person_to_train]", "zed"))
    del config["model"], config["dataset"]["path"], config["output"]["path"]
    return config


def bria_run(ws: Workspace, monkeypatch: pytest.MonkeyPatch, steps: int) -> tuple[Fibo2LoRATrainer, int]:
    """The template on the tiny world (32 px, fp32, ``steps`` optimizer steps); the trainer and its micro-steps."""
    ws.add_images(3)
    samples = 0
    step = Fibo2LoRATrainer._training_step

    def counted(self: Fibo2LoRATrainer, index: int) -> torch.Tensor:
        nonlocal samples
        samples += 1
        return step(self, index)

    monkeypatch.setattr(Fibo2LoRATrainer, "_training_step", counted)
    config = ws.config(
        bria_sections(), dataset={"resolution": 32}, train={"max_steps": steps}, dtype="float32", mixed_precision="no"
    )
    return run_lora(config), samples


def test_bria_template_trains_adamw_at_1e_4_with_weight_decay_and_a_constant_rate(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, _ = bria_run(workspace, monkeypatch, steps=1)

    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    assert type(optimizer) is torch.optim.AdamW
    group = optimizer.param_groups[0]
    assert (group["lr"], group["weight_decay"], group["betas"], group["eps"]) == (1e-4, 0.01, (0.9, 0.999), 1e-8)
    assert (trainer.lr_scheduler, trainer.ema) == (None, None)


def test_bria_template_accumulates_eight_samples_per_optimizer_step(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, samples = bria_run(workspace, monkeypatch, steps=2)

    optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
    states = [optimizer.state[p] for p in trainer.lora_network.trainable_parameters()]
    assert samples == 16
    assert {int(state["step"]) for state in states} == {2}


def test_bria_template_adapts_every_linear_at_rank_64_and_saves_fp32(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer, _ = bria_run(workspace, monkeypatch, steps=1)

    saved = load_file(str(workspace.out / "fibo2_bria_zed.safetensors"))
    linears = {name for name, module in trainer.transformer.named_modules() if isinstance(module, nn.Linear)}
    adapted = {key.removeprefix("diffusion_model.").removesuffix(".lora_A.weight") for key in saved if "lora_A" in key}
    assert adapted == linears
    assert {saved[key].shape[0] for key in saved if key.endswith("lora_A.weight")} == {64}
    assert {float(saved[key]) for key in saved if key.endswith(".alpha")} == {64.0}
    assert {saved[key].dtype for key in saved if "lora_" in key} == {torch.float32}


def test_bria_template_leaves_the_sigma_draw_unclamped() -> None:
    train = bria_sections()["train"]

    assert clamped(torch.tensor([0.0, 1.0]), train).tolist() == [0.0, 1.0]
    assert clamped(torch.tensor([0.0, 1.0]), {}).tolist() == pytest.approx([0.001, 0.999])
