"""Options of the Z-Image trainers that used to crash or be ignored: the Turbo training adapter, batches of
more than one, save_every 0 and a null trigger."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import train_zimage_embedding as embedding
import train_zimage_lora as lora

from .fakes import Fakes
from .runs import run_embedding, run_lora
from .tiny import build_transformer
from .world import Workspace, named_weights

Q = "layers.0.attention.to_q"


def _adapter_file(path: Path) -> Path:
    gen = torch.Generator().manual_seed(3)
    tensors = {
        f"diffusion_model.{Q}.lora_A.weight": torch.randn(2, 16, generator=gen),
        f"diffusion_model.{Q}.lora_B.weight": torch.randn(16, 2, generator=gen),
    }
    save_file(tensors, str(path))
    return path


def test_the_turbo_training_adapter_is_merged_while_the_model_loads(
    workspace: Workspace, fakes: Fakes, tmp_path: Path
) -> None:
    adapter = _adapter_file(tmp_path / "adapter.safetensors")
    trainer = lora.ZImageLoRATrainer(workspace.lora_config(model={"assistant_lora_path": str(adapter)}))
    base = named_weights(build_transformer())

    trainer.load_model()

    after = named_weights(trainer.transformer)
    assert [name for name in base if not torch.equal(base[name], after[name])] == [f"{Q}.weight"]
    assert fakes.loads


@pytest.mark.parametrize("kind", ["lora", "embedding"])
def test_a_batch_of_two_trains_instead_of_failing_on_the_first_sample_only(workspace: Workspace, kind: str) -> None:
    workspace.add_images(4)
    train = {"batch_size": 2, "num_epochs": 1}

    if kind == "lora":
        trainer: lora.ZImageLoRATrainer | embedding.ZImageEmbeddingTrainer = run_lora(
            workspace.lora_config(train=train)
        )
    else:
        trainer = run_embedding(workspace.embedding_config(train=train))

    assert sorted(path.suffix for path in workspace.out.glob("*.safetensors")) == [".safetensors"]
    assert trainer.accelerator.sync_gradients


@pytest.mark.parametrize("kind", ["lora", "embedding"])
def test_save_every_zero_means_only_the_final_file(workspace: Workspace, kind: str) -> None:
    workspace.add_images(2)
    train = {"num_epochs": 2, "save_every": 0}

    if kind == "lora":
        run_lora(workspace.lora_config(train=train))
    else:
        run_embedding(workspace.embedding_config(train=train))

    assert len(list(workspace.out.glob("*.safetensors"))) == 1


def test_a_null_dataset_trigger_means_no_trigger(workspace: Workspace) -> None:
    workspace.add_images(2)
    config = workspace.lora_config(train={"num_epochs": 1})
    config["dataset"]["trigger"] = None

    run_lora(config)

    assert len(list(workspace.out.glob("*.safetensors"))) == 1
