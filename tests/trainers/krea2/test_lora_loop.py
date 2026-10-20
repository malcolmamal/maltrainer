"""The training loop: how many steps, epochs and saves a config means, and what each step trains on."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from maltrainer.models.krea2.lora import Krea2LoRANetwork
from train_krea2_lora import Krea2LoRATrainer, plan_training_steps

from .runs import prepared_lora, progress_lines, run_lora
from .world import Workspace, named_weights, same_values


@pytest.mark.parametrize(
    ("samples", "train", "expected"),
    [
        (5, {}, (5, 10, 50)),
        (5, {"max_steps": 12}, (5, 3, 12)),
        (5, {"gradient_accumulation_steps": 2, "num_epochs": 4}, (3, 4, 10)),
        (3, {"gradient_accumulation_steps": 2, "max_steps": 4}, (2, 3, 4)),
        (0, {"max_steps": 3}, (1, 3, 3)),
        (4, {"num_epochs": 0, "max_steps": None}, (4, 1, 4)),
    ],
)
def test_the_plan_lets_max_steps_win_over_epochs(samples: int, train: dict[str, Any], expected: Any) -> None:
    assert plan_training_steps(samples, train) == expected


def test_max_steps_ends_the_run_in_the_middle_of_an_epoch(workspace: Workspace) -> None:
    workspace.add_images(2)

    trainer = run_lora(workspace.lora_config(train={"max_steps": 3}))

    lines = progress_lines(workspace)
    assert len(lines) == 3
    assert "Step: 3/3 [100%]" in lines[-1] and "Epoch: 2/2" in lines[-1]
    assert trainer.run_metadata["planned_total_steps"] == 3
    assert trainer.run_metadata["steps_per_epoch"] == 2


def test_accumulation_over_a_sample_count_it_does_not_divide_still_takes_every_planned_step(
    workspace: Workspace,
) -> None:
    workspace.add_images(3)

    run_lora(workspace.lora_config(train={"max_steps": 4, "gradient_accumulation_steps": 2}))

    lines = progress_lines(workspace)
    assert len(lines) == 4
    assert "Step: 4/4 [100%]" in lines[-1]


def test_without_max_steps_the_epochs_decide(workspace: Workspace) -> None:
    workspace.add_images(2)

    run_lora(workspace.lora_config(train={"max_steps": 0, "num_epochs": 2}))

    assert len(progress_lines(workspace)) == 4


def test_gradient_accumulation_steps_the_optimizer_once_per_group_of_samples(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    calls: list[int] = []
    original = Krea2LoRATrainer._training_step

    def counting(self: Krea2LoRATrainer, index: int, shift_logged: list[bool]) -> tuple[torch.Tensor, ...]:
        calls.append(index)
        return original(self, index, shift_logged)

    monkeypatch.setattr(Krea2LoRATrainer, "_training_step", counting)

    trainer = run_lora(workspace.lora_config(train={"max_steps": 2, "gradient_accumulation_steps": 2}))

    assert len(calls) == 4
    assert len(progress_lines(workspace)) == 2
    assert trainer.run_metadata["steps_per_epoch"] == 1


def test_save_every_writes_numbered_files_but_never_one_for_the_final_step(workspace: Workspace) -> None:
    workspace.add_images(2)

    run_lora(workspace.lora_config(train={"max_steps": 3, "save_every": 1}))

    assert sorted(path.name for path in workspace.out.glob("*.safetensors")) == [
        "k2.safetensors",
        "k2_step_1.safetensors",
        "k2_step_2.safetensors",
    ]


def test_the_same_seed_gives_bit_identical_adapters(workspace: Workspace) -> None:
    workspace.add_images(3)
    config = workspace.lora_config(train={"max_steps": 4, "caption_dropout": 0.5, "timestep_distribution": "uniform"})

    first = run_lora(config)
    second = run_lora(workspace.lora_config(output={"path": str(workspace.root / "again")}, train=config["train"]))

    assert same_values(named_weights(first.lora_network), named_weights(second.lora_network))


def test_caption_dropout_one_trains_every_step_on_the_empty_prompt(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_images(2)
    contexts: list[torch.Tensor] = []
    original = Krea2LoRANetwork.forward

    def spy(self: Krea2LoRANetwork, *args: Any, **kwargs: Any) -> Any:
        contexts.append(args[2][0])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Krea2LoRANetwork, "forward", spy)

    trainer = run_lora(workspace.lora_config(train={"caption_dropout": 1.0}))

    assert len(contexts) == 3
    assert all(torch.equal(context, trainer.text_cache[""]) for context in contexts)


def test_the_loss_is_mean_squared_error_against_noise_minus_latent_scaled_by_the_strength(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared_lora(workspace.lora_config())

    torch.manual_seed(3)
    plain = trainer._training_step(0, [True])[0]
    trainer.config["train"]["loss_weight_strength"] = 3.0
    torch.manual_seed(3)
    scaled = trainer._training_step(0, [True])[0]

    assert plain.item() > 0
    torch.testing.assert_close(scaled, 3 * plain)


@pytest.mark.parametrize(("repeats", "copies"), [(1, 1), (3, 3), (0, 1)])
def test_an_epoch_visits_every_sample_repeats_times(workspace: Workspace, repeats: int, copies: int) -> None:
    workspace.add_images(3)
    trainer = Krea2LoRATrainer(workspace.lora_config(dataset={"repeats": repeats}))
    trainer.samples = [(str(i), "") for i in range(3)]

    order = trainer._epoch_order()

    assert sorted(order) == sorted(list(range(3)) * copies)


def test_with_several_processes_each_takes_its_own_slice_of_the_shuffled_epoch(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = Krea2LoRATrainer(workspace.lora_config())
    trainer.samples = [(str(i), "") for i in range(6)]
    random.seed(0)
    whole = trainer._epoch_order()

    monkeypatch.setattr(trainer, "accelerator", SimpleNamespace(num_processes=2, process_index=1))
    random.seed(0)

    assert trainer._epoch_order() == whole[1::2]
