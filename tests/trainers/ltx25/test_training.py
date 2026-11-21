"""A LoRA run on the tiny DiT: only the adapters train, and the saved file is the LoRA ComfyUI loads."""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from maltrainer.utils.lora_metadata import read_metadata
from train_ltx25_lora import LTX25LoRATrainer

from .runs import prepared, run_lora
from .world import Workspace, base_weights, named_weights, same_values


def trained(ws: Workspace, **sections: Any) -> tuple[LTX25LoRATrainer, dict[str, torch.Tensor]]:
    """A run on two clips with sound and a still."""
    ws.add_clip("clip0", seconds=0.5)
    ws.add_clip("clip1", seconds=0.5, caption="zed runs")
    ws.add_images(1)
    trainer = run_lora(ws.config(**sections))
    return trainer, load_file(str(ws.out / "ltx25.safetensors"))


def test_only_the_adapters_train_and_the_base_stays_bit_identical(workspace: Workspace) -> None:
    trainer, saved = trained(workspace)

    assert same_values(named_weights(trainer.transformer), base_weights(workspace))
    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    assert all(torch.count_nonzero(saved[key]) > 0 for key in saved if key.endswith("lora_B.weight"))


def test_the_saved_file_holds_an_a_b_and_alpha_per_adapter_in_the_output_dtype(workspace: Workspace) -> None:
    _, saved = trained(workspace)

    assert len(saved) == 3 * 48
    assert saved["diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight"].shape == (4, 96)
    assert saved["diffusion_model.transformer_blocks.0.audio_to_video_attn.to_out.0.lora_B.weight"].shape == (96, 4)
    assert saved["diffusion_model.transformer_blocks.1.audio_attn2.to_k.lora_A.weight"].shape == (4, 48)
    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.bfloat16}
    assert {saved[k].item() for k in saved if k.endswith("alpha")} == {4.0}


def test_a_still_only_dataset_saves_a_video_only_lora(workspace: Workspace) -> None:
    workspace.add_images(2)
    run_lora(workspace.config())

    saved = load_file(str(workspace.out / "ltx25.safetensors"))

    assert len(saved) == 3 * 16 and not any("audio" in key for key in saved)


def test_rank_alpha_and_prefix_shape_the_adapters(workspace: Workspace) -> None:
    _, saved = trained(workspace, lora={"rank": 8, "alpha": 2.5, "prefix": "unet.", "targets": "video"})

    assert {key.split(".")[0] for key in saved} == {"unet"}
    assert saved["unet.transformer_blocks.1.attn1.to_k.lora_A.weight"].shape == (8, 96)
    assert saved["unet.transformer_blocks.1.attn1.to_k.alpha"].item() == 2.5
    assert not any("audio" in key for key in saved)


def test_a_clip_with_sound_scores_both_streams_and_a_still_only_the_video(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace.add_clip("clip0", seconds=0.5)
    workspace.add_images(1)
    trainer = prepared(workspace.config())
    seen: list[torch.Tensor | None] = []
    original = trainer.lora_network.forward

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(args[4])
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer.lora_network, "forward", spy)
    index = {name.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]: i for i, (name, _) in enumerate(trainer.samples)}

    trainer._training_step(index["clip0.avi"])
    trainer._training_step(index["img0.png"])

    assert seen[0] is not None and seen[0].shape == (1, 8, 9, 16) and seen[1] is None


def test_the_loss_is_finite_and_differs_between_steps(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared(workspace.config())

    losses = [trainer._training_step(0).item() for _ in range(3)]

    assert all(torch.isfinite(torch.tensor(losses))) and len(set(losses)) == 3


def test_metadata_records_the_architecture_the_adapters_and_the_config(workspace: Workspace) -> None:
    trainer, _ = trained(workspace, train={"ema_decay": 0.5})

    meta = read_metadata(workspace.out / "ltx25.safetensors")
    assert (meta["maltrainer_architecture"], meta["maltrainer_trainer"]) == ("ltx25", "train_ltx25_lora.py")
    assert meta["ss_network_module"] == "maltrainer.models.ltx25.lora"
    assert (meta["ss_network_dim"], meta["ss_network_alpha"], meta["step"]) == ("4", "4", "final")
    assert (meta["maltrainer_lora_layers"], meta["maltrainer_lora_dtype"]) == ("48", "float32")
    assert json.loads(meta["maltrainer_optimizer_resolved"])["name"] == "automagic3"
    assert int(meta["maltrainer_trainable_parameters"]) == trainer.run_metadata["trainable_parameters"]


def test_with_ema_the_checkpoint_holds_the_average_and_the_raw_adapters_come_back(workspace: Workspace) -> None:
    trainer, saved = trained(workspace, train={"ema_decay": 0.5, "ema_warmup": False})

    assert trainer.ema is not None
    network = trainer.accelerator.unwrap_model(trainer.lora_network)
    name = "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight"
    assert not torch.equal(network.get_state_dict(torch.bfloat16)[name], saved[name])
    with trainer.ema.applied():
        assert torch.equal(network.get_state_dict(torch.bfloat16)[name], saved[name])


@pytest.mark.parametrize("optimizer", ["adamw8bit", "prodigy"])
def test_the_other_optimizers_run_too(workspace: Workspace, optimizer: str) -> None:
    workspace.add_images(2)

    run_lora(workspace.config(train={"optimizer": optimizer}))

    assert (workspace.out / "ltx25.safetensors").exists()
