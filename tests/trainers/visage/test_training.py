"""A LoRA run on the tiny DiT: only the adapters train, one step follows the model's recipe, and the saved file is the
LoRA the other trainers' tools read."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import pytest
import torch

from maltrainer.utils.lora_metadata import read_metadata

from .runs import prepared, run_lora, trained
from .world import Workspace, base_weights, named_weights, same_values

FIXED_SIGMA = {"timestep_distribution": "logit_normal", "timestep_logit_std": 0.0, "timestep_logit_mean": 0.0}


def _modules(saved: dict[str, torch.Tensor]) -> set[str]:
    return {key.removeprefix("diffusion_model.").rsplit(".", 2)[0] for key in saved if key.endswith("lora_A.weight")}


def test_only_the_adapters_train_and_the_base_stays_bit_identical(workspace: Workspace) -> None:
    trainer, saved = trained(workspace)

    assert same_values(named_weights(trainer.transformer), base_weights(workspace))
    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    assert all(torch.count_nonzero(saved[key]) > 0 for key in saved if key.endswith("lora_B.weight"))


def test_the_saved_file_holds_an_a_b_and_alpha_per_adapter_in_the_output_dtype(workspace: Workspace) -> None:
    _, saved = trained(workspace)

    assert len(saved) == 3 * 8
    assert saved["diffusion_model.blocks.0.attn.qkv.lora_A.weight"].shape == (4, 128)
    assert saved["diffusion_model.blocks.0.attn.qkv.lora_B.weight"].shape == (3 * 128, 4)
    assert saved["diffusion_model.blocks.1.mlp.fc2.lora_A.weight"].shape == (4, 2 * 128)
    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.bfloat16}
    assert {saved[k].item() for k in saved if k.endswith("alpha")} == {4.0}


@pytest.mark.parametrize(("targets", "count"), [("blocks", 8), ("attn", 4), ("all", 15)])
def test_lora_targets_picks_which_linears_get_an_adapter(workspace: Workspace, targets: str, count: int) -> None:
    _, saved = trained(workspace, lora={"targets": targets})

    assert len(_modules(saved)) == count
    assert ("text_proj" in _modules(saved)) is (targets == "all")


def test_an_unknown_target_preset_is_refused(workspace: Workspace) -> None:
    workspace.add_images(1)

    with pytest.raises(ValueError, match=r"lora\.targets must be one of \('blocks', 'attn', 'all'\), got 'mlp'"):
        run_lora(workspace.config(lora={"targets": "mlp"}))


def test_rank_alpha_and_prefix_shape_the_adapters(workspace: Workspace) -> None:
    _, saved = trained(workspace, lora={"rank": 8, "alpha": 2.5, "prefix": "unet."})

    assert {key.split(".")[0] for key in saved} == {"unet"}
    assert saved["unet.blocks.1.attn.out_proj.lora_A.weight"].shape == (8, 128)
    assert saved["unet.blocks.1.attn.out_proj.alpha"].item() == 2.5


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

    meta = read_metadata(workspace.out / "visage.safetensors")
    assert (meta["maltrainer_architecture"], meta["maltrainer_trainer"]) == ("visage", "train_visage_lora.py")
    assert meta["ss_network_module"] == "maltrainer.models.visage.lora"
    assert meta["ss_base_model_version"] == "visage_image"
    assert (meta["ss_network_dim"], meta["ss_network_alpha"], meta["step"]) == ("4", "4", "final")
    assert (meta["maltrainer_lora_layers"], meta["maltrainer_lora_dtype"]) == ("8", "float32")
    assert json.loads(meta["maltrainer_optimizer_resolved"])["name"] == "automagic3"
    assert int(meta["maltrainer_trainable_parameters"]) == trainer.run_metadata["trainable_parameters"]


def test_with_ema_the_checkpoint_holds_the_average_and_the_raw_adapters_come_back(workspace: Workspace) -> None:
    trainer, saved = trained(workspace, train={"ema_decay": 0.5, "ema_warmup": False})

    assert trainer.ema is not None
    network = trainer.accelerator.unwrap_model(trainer.lora_network)
    name = "diffusion_model.blocks.0.attn.qkv.lora_B.weight"
    assert not torch.equal(network.get_state_dict(torch.bfloat16)[name], saved[name])
    with trainer.ema.applied():
        assert torch.equal(network.get_state_dict(torch.bfloat16)[name], saved[name])


def test_the_start_line_names_the_release_timestep_recipe(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        trained(workspace, 1)

    assert "Starting Visage-Image LoRA training: " in caplog.text and "timesteps visage, shift 1.00" in caplog.text


class Spy:
    """Wraps the LoRA network's forward and keeps what the trainer fed the DiT and what it answered."""

    def __init__(self, network: torch.nn.Module) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        original = network.forward

        def forward(x: torch.Tensor, t: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
            out: torch.Tensor = original(x, t, text)
            self.calls.append((x.clone(), t.clone(), text.clone(), out.detach().clone()))
            return out

        network.forward = forward  # type: ignore[method-assign]


def _step(workspace: Workspace, **train: Any) -> tuple[Any, Spy, torch.Tensor]:
    workspace.add_images(1)
    trainer = prepared(workspace.config(train=train))
    spy = Spy(trainer.lora_network)
    return trainer, spy, trainer._training_step(0)


def test_a_step_noises_to_sigma_and_regresses_the_velocity_noise_minus_data(workspace: Workspace) -> None:
    trainer, spy, loss = _step(workspace, **FIXED_SIGMA)

    x, t, text, pred = spy.calls[0]
    x0 = trainer.latent_cache[trainer.samples[0][0]].unsqueeze(0)
    assert t.shape == (1, 1) and t.item() == pytest.approx(0.5)
    noise = (x - 0.5 * x0) / 0.5  # x_t = (1 - sigma) x0 + sigma noise
    assert noise.std() > 0.3  # N(0, 1) noise, not a copy of the data
    assert loss.item() == pytest.approx(torch.nn.functional.mse_loss(pred, noise - x0).item(), rel=1e-5)
    assert torch.equal(text, trainer.text_cache["a photo of zed"].unsqueeze(0))


def test_the_timestep_is_the_release_logit_normal_around_the_token_shift(workspace: Workspace) -> None:
    trainer, spy, _ = _step(workspace, timestep_logit_std=0.0)

    assert spy.calls[0][1].item() == pytest.approx(1 / (1 + math.exp(-0.5 * math.log(4 / 64))))  # 4 latent cells
    assert trainer.last_sigma == pytest.approx(spy.calls[0][1].item())


def test_every_loss_is_finite_and_the_step_trains_the_adapters_only(workspace: Workspace) -> None:
    trainer, _, loss = _step(workspace)

    loss.backward()

    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in trainer.lora_network.trainable_parameters())
    assert all(p.grad is None for p in trainer.transformer.parameters())


def test_without_dropout_the_caption_states_reach_the_dit(workspace: Workspace) -> None:
    _, spy, _ = _step(workspace, caption_dropout=0.0)

    assert torch.count_nonzero(spy.calls[0][2]) > 0


def test_a_dropped_caption_becomes_zeros_of_its_own_length(workspace: Workspace) -> None:
    trainer, spy, _ = _step(workspace, caption_dropout=1.0)

    text = spy.calls[0][2]
    assert text.shape == trainer.text_cache["a photo of zed"].unsqueeze(0).shape and text.abs().max() == 0


def test_the_dropout_rate_is_the_share_of_captions_dropped(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared(workspace.config(train={"caption_dropout": 0.3}))
    spy = Spy(trainer.lora_network)
    torch.manual_seed(1)

    for _ in range(400):
        trainer._training_step(0)

    dropped = sum(1 for call in spy.calls if call[2].abs().max() == 0)
    assert 0.2 < dropped / 400 < 0.4
