"""Qwen-Image 2.1 LoRA options that used to be ignored, silently wrong or fail late."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import train_qwen_image_2_1_lora as qwen

from .runs import loaded, prepared, run_qwen
from .world import Workspace


def test_sigmas_are_drawn_in_float32_even_when_the_run_trains_in_bfloat16(workspace: Workspace) -> None:
    trainer = qwen.QwenImage21LoRATrainer(workspace.config(dtype="bfloat16"))

    drawn = trainer._sample_timesteps(512)

    assert drawn.dtype == torch.float32
    assert (drawn.to(torch.bfloat16).float() != drawn).any()  # not snapped to bfloat16's grid


def test_a_bfloat16_run_still_feeds_the_network_bfloat16_hidden_states(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    trainer = prepared(workspace.config(dtype="bfloat16", train={"max_steps": 1}))
    seen: list[torch.dtype] = []
    real = trainer.lora_network.forward

    def noting(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["hidden_states"].dtype)
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer.lora_network, "forward", noting)

    trainer.train()

    assert seen == [torch.bfloat16]


@pytest.mark.parametrize("key", ["warmup_steps", "learning_rate_warmup_steps"])
def test_either_warmup_key_configures_a_warmup(workspace: Workspace, key: str) -> None:
    workspace.add_images(2)

    trainer = prepared(workspace.config(train={key: 5}))

    assert trainer.lr_scheduler is not None


@pytest.mark.parametrize("key", ["warmup_steps", "learning_rate_warmup_steps"])
def test_a_null_warmup_means_none(workspace: Workspace, key: str) -> None:
    workspace.add_images(2)

    trainer = prepared(workspace.config(train={key: None}))

    assert trainer.lr_scheduler is None


def test_save_every_zero_means_only_the_final_file(workspace: Workspace) -> None:
    workspace.add_images(2)

    run_qwen(workspace.config(train={"max_steps": 3, "save_every": 0}))

    assert [path.name for path in workspace.out.glob("*.safetensors")] == ["qwen_image_2_1_lora.safetensors"]


def test_a_batch_of_captions_with_different_lengths_is_refused_clearly(workspace: Workspace) -> None:
    workspace.add_images(1, caption="one", folder="a")
    workspace.add_images(1, caption="one two three four", folder="b")

    with pytest.raises(ValueError, match=r"different token lengths.*batch_size 1.*gradient_accumulation_steps"):
        run_qwen(workspace.config(train={"batch_size": 2, "max_steps": 1}))


def test_lora_targets_that_match_nothing_are_refused_at_once(workspace: Workspace) -> None:
    workspace.add_images(2)

    with pytest.raises(ValueError, match=r"No Qwen-Image 2\.1 module matched lora\.target_modules \['nope'\]"):
        loaded(workspace.config(lora={"target_modules": ["nope"]}))


def test_a_checkpoint_is_written_from_the_unwrapped_network(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    trainer = prepared(workspace.config())
    asked: list[object] = []
    real = trainer.accelerator.unwrap_model

    def unwrap(model: Any, *args: Any, **kwargs: Any) -> Any:
        asked.append(model)
        return real(model, *args, **kwargs)

    monkeypatch.setattr(trainer.accelerator, "unwrap_model", unwrap)

    trainer.save_checkpoint("final")

    assert asked == [trainer.lora_network]
