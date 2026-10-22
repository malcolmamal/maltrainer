"""A LoRA run on the tiny model: only the adapters train, and the saved file holds exactly them."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import train_zimage_lora as lora

from .fakes import Fakes
from .runs import run_lora
from .tiny import build_text_encoder, build_transformer, build_vae
from .world import Workspace, changed, lora_layers, named_weights

TARGETS = ("to_q", "to_k", "to_v", "to_out")


def _trained(ws: Workspace, images: int = 3, **sections: Any) -> lora.ZImageLoRATrainer:
    ws.add_images(images)
    return run_lora(ws.lora_config(**sections))


def _target_names() -> list[str]:
    modules = build_transformer().named_modules()
    return [n for n, m in modules if isinstance(m, torch.nn.Linear) and any(t in n for t in TARGETS)]


def test_only_the_adapters_train_and_every_base_model_stays_bit_identical(workspace: Workspace, fakes: Fakes) -> None:
    trainer = _trained(workspace)

    assert not changed(named_weights(build_transformer()), named_weights(trainer.transformer))
    assert not changed(named_weights(build_text_encoder()), named_weights(fakes.parts["text_encoder"]))
    assert not changed(named_weights(build_vae()), named_weights(fakes.parts["vae"]))
    assert all(torch.count_nonzero(layer.lora_up.weight) > 0 for layer in lora_layers(trainer))


def test_the_saved_file_holds_one_a_b_and_alpha_per_targeted_linear(workspace: Workspace) -> None:
    _trained(workspace)

    saved = load_file(str(workspace.out / "zimage_lora_step_final.safetensors"))
    names = _target_names()
    assert len(names) == 12
    assert set(saved) == {f"{n}.{part}" for n in names for part in ("lora_A.weight", "lora_B.weight", "alpha")}
    assert saved[f"{names[0]}.lora_A.weight"].shape == (4, 16)
    assert saved[f"{names[0]}.lora_B.weight"].shape[1] == 4
    assert {saved[f"{n}.alpha"].item() for n in names} == {4.0}


def test_the_checkpoint_metadata_records_step_rank_and_alpha(workspace: Workspace) -> None:
    _trained(workspace)

    with safe_open(str(workspace.out / "zimage_lora_step_final.safetensors"), framework="pt") as handle:
        assert handle.metadata() == {"step": "final", "rank": "4", "alpha": "4"}


def test_a_config_without_a_lora_section_uses_rank_4_and_alpha_1(workspace: Workspace) -> None:
    workspace.add_images(1)
    config = workspace.lora_config()
    del config["lora"]

    trainer = run_lora(config)

    with safe_open(str(workspace.out / "zimage_lora_step_final.safetensors"), framework="pt") as handle:
        assert handle.metadata() == {"step": "final", "rank": "4", "alpha": "1.0"}
    assert (trainer.lora_network.rank, trainer.lora_network.alpha) == (4, 1.0)


def test_a_checkpoint_is_written_every_save_every_steps_and_one_at_the_end(workspace: Workspace) -> None:
    trainer = _trained(workspace, images=3, train={"num_epochs": 2, "save_every": 2})

    names = sorted(p.name for p in workspace.out.glob("*.safetensors"))
    assert names == [f"zimage_lora_step_{step}.safetensors" for step in (2, 4, 6, "final")]
    assert trainer.last_saved_path == str(workspace.out / "zimage_lora_step_final.safetensors")


def test_each_step_is_logged_with_its_loss_epoch_and_learning_rate(workspace: Workspace) -> None:
    _trained(workspace, images=2, train={"num_epochs": 2})

    lines = [x for x in (workspace.out / "training.log").read_text(encoding="utf-8").splitlines() if "Step:" in x]
    assert [x.split(" - progress - INFO - ")[1].split(", Loss")[0] for x in lines] == [
        "Step: 1/4 [25%]",
        "Step: 2/4 [50%]",
        "Step: 3/4 [75%]",
        "Step: 4/4 [100%]",
    ]
    assert [x.rsplit(", Epoch: ", 1)[1] for x in lines] == [f"{e}/2, LR: 1.00e-03" for e in (1, 1, 2, 2)]


def test_the_transformer_is_fed_noisy_latents_the_flipped_timestep_and_the_text_hidden_states(
    workspace: Workspace, fakes: Fakes
) -> None:
    _trained(workspace, images=1)

    call = fakes.calls[0]
    assert call.x.shape == (1, 4, 1, 8, 8)
    assert call.t.shape == (1,) and 0.0 < call.t.item() <= 1.0
    assert call.cap_feats.shape[0] == 1 and call.cap_feats.shape[2] == 16
    assert call.cap_mask is None


def test_the_training_banner_names_the_device_dtype_and_model_sizes(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    _trained(workspace, images=1)

    lines = capsys.readouterr().out.splitlines()
    assert lines[:5] == [
        "--- Training Configuration ---",
        "Device: CPU",
        "Dtype: torch.float32",
        "Transformer: 0.00B params",
        "Text Encoder: 0.00B params",
    ]


def test_the_same_seed_gives_the_same_adapters_and_another_seed_does_not(workspace: Workspace) -> None:
    def adapters(seed: int) -> dict[str, torch.Tensor]:
        trainer = _trained(workspace, seed=seed)
        return named_weights(trainer.lora_network.lora_layers)

    first, again, other = adapters(7), adapters(7), adapters(8)

    assert not changed(first, again)
    assert changed(first, other)
