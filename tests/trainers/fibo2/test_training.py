"""A LoRA run on the tiny DiT: only the adapters train, and the saved file is the LoRA ComfyUI loads."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from maltrainer.utils.lora_metadata import read_metadata
from train_fibo2_lora import Fibo2LoRATrainer

from .runs import run_lora
from .world import Workspace, base_weights, named_weights, same_values


def trained(ws: Workspace, images: int = 2, **sections: Any) -> tuple[Fibo2LoRATrainer, dict[str, torch.Tensor]]:
    ws.add_images(images)
    trainer = run_lora(ws.config(**sections))
    return trainer, load_file(str(ws.out / "fibo2.safetensors"))


def _modules(saved: dict[str, torch.Tensor]) -> set[str]:
    return {key.removeprefix("diffusion_model.").rsplit(".", 2)[0] for key in saved if key.endswith("lora_A.weight")}


def test_only_the_adapters_train_and_the_base_stays_bit_identical(workspace: Workspace) -> None:
    trainer, saved = trained(workspace)

    assert same_values(named_weights(trainer.transformer), base_weights(workspace))
    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    assert all(torch.count_nonzero(saved[key]) > 0 for key in saved if key.endswith("lora_B.weight"))


def test_the_saved_file_holds_an_a_b_and_alpha_per_adapter_in_the_output_dtype(workspace: Workspace) -> None:
    _, saved = trained(workspace)

    assert len(saved) == 3 * 43
    assert saved["diffusion_model.layers.0.attention.qkv.lora_A.weight"].shape == (4, 256)
    assert saved["diffusion_model.layers.0.attention.qkv.lora_B.weight"].shape == (768, 4)
    assert saved["diffusion_model.layers.1.cross_attention.to_kv.lora_A.weight"].shape == (4, 16)
    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.bfloat16}
    assert {saved[k].item() for k in saved if k.endswith("alpha")} == {4.0}


@pytest.mark.parametrize(("targets", "count"), [("blocks", 43), ("main", 39), ("all", 63)])
def test_lora_targets_picks_which_linears_get_an_adapter(workspace: Workspace, targets: str, count: int) -> None:
    _, saved = trained(workspace, lora={"targets": targets})

    assert len(_modules(saved)) == count
    assert ("x_embedder" in _modules(saved)) is (targets == "all")


def test_an_unknown_target_preset_is_refused(workspace: Workspace) -> None:
    workspace.add_images(1)

    with pytest.raises(ValueError, match=r"lora\.targets must be one of \('blocks', 'main', 'all'\), got 'mlp'"):
        run_lora(workspace.config(lora={"targets": "mlp"}))


def test_rank_alpha_and_prefix_shape_the_adapters(workspace: Workspace) -> None:
    _, saved = trained(workspace, lora={"rank": 8, "alpha": 2.5, "prefix": "unet."})

    assert {key.split(".")[0] for key in saved} == {"unet"}
    assert saved["unet.layers.1.attention.out.lora_A.weight"].shape == (8, 256)
    assert saved["unet.layers.1.attention.out.alpha"].item() == 2.5


def test_the_alpha_defaults_to_the_rank_and_the_lora_dtype_is_independent_of_the_output(workspace: Workspace) -> None:
    config = workspace.config(output={"dtype": "float32"})
    del config["lora"]["alpha"]
    workspace.add_images(1)

    trainer = run_lora(config)

    assert trainer.run_metadata["lora_dtype"] == "float32"
    assert {p.dtype for p in trainer.lora_network.trainable_parameters()} == {torch.float32}
    saved = trainer.lora_network.get_state_dict(None)
    assert {saved[k].item() for k in saved if k.endswith("alpha")} == {4.0}


def test_metadata_records_the_architecture_the_adapters_and_the_config(workspace: Workspace) -> None:
    trainer, _ = trained(workspace, train={"ema_decay": 0.5})

    meta = read_metadata(workspace.out / "fibo2.safetensors")
    assert (meta["maltrainer_architecture"], meta["maltrainer_trainer"]) == ("fibo2", "train_fibo2_lora.py")
    assert meta["ss_network_module"] == "maltrainer.models.fibo2.lora"
    assert (meta["ss_network_dim"], meta["ss_network_alpha"], meta["step"]) == ("4", "4", "final")
    assert (meta["maltrainer_lora_layers"], meta["maltrainer_lora_dtype"]) == ("43", "float32")
    assert json.loads(meta["maltrainer_optimizer_resolved"])["name"] == "automagic3"
    assert int(meta["maltrainer_trainable_parameters"]) == trainer.run_metadata["trainable_parameters"]


def test_with_ema_the_checkpoint_holds_the_average_and_the_raw_adapters_come_back(workspace: Workspace) -> None:
    trainer, saved = trained(workspace, train={"ema_decay": 0.5, "ema_warmup": False})

    assert trainer.ema is not None
    network = trainer.accelerator.unwrap_model(trainer.lora_network)
    name = "diffusion_model.layers.0.attention.qkv.lora_B.weight"
    assert not torch.equal(network.get_state_dict(torch.bfloat16)[name], saved[name])
    with trainer.ema.applied():
        assert torch.equal(network.get_state_dict(torch.bfloat16)[name], saved[name])
