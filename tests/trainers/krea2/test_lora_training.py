"""A LoRA run on the tiny Krea 2 DiT: only the adapters train, and the saved file is the LoRA ComfyUI loads."""

from __future__ import annotations

import json

import pytest
import torch

from maltrainer.models.krea2.loading import load_krea2_transformer
from maltrainer.utils.lora_metadata import read_metadata

from .runs import run_lora, trained_lora
from .world import Workspace, named_weights, same_values, write_checkpoint

BLOCK_LEAVES = ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo", "mlp.gate", "mlp.up", "mlp.down")
IO_MODULES = {"first", "tmlp.0", "tmlp.2", "txtmlp.1", "txtmlp.3", "tproj.1", "last.linear"}


def _modules(saved: dict[str, torch.Tensor]) -> set[str]:
    return {key.removeprefix("diffusion_model.").rsplit(".", 2)[0] for key in saved if key.endswith("lora_A.weight")}


def _blocks(*indices: int) -> set[str]:
    return {f"blocks.{i}.{leaf}" for i in indices for leaf in BLOCK_LEAVES}


def _comparable(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.float() if v.is_floating_point() else v for k, v in state.items()}


def test_only_the_adapters_train_and_the_base_stays_bit_identical(workspace: Workspace) -> None:
    trainer, saved = trained_lora(workspace)

    fresh, _ = load_krea2_transformer(workspace.checkpoint, torch.device("cpu"), torch.float32)
    assert same_values(_comparable(named_weights(trainer.transformer)), _comparable(named_weights(fresh)))
    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    assert all(torch.count_nonzero(saved[key]) > 0 for key in saved if key.endswith("lora_B.weight"))


def test_the_saved_file_holds_an_a_b_and_alpha_per_adapter_in_the_output_dtype(workspace: Workspace) -> None:
    _, saved = trained_lora(workspace)

    assert len(saved) == 3 * 16
    assert saved["diffusion_model.blocks.0.attn.wq.lora_A.weight"].shape == (4, 64)
    assert saved["diffusion_model.blocks.1.mlp.down.lora_B.weight"].shape == (64, 4)
    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.bfloat16}
    assert {saved[k].item() for k in saved if k.endswith("alpha")} == {4.0}


@pytest.mark.parametrize(
    ("targets", "count", "text_fusion", "io_layers"),
    [("attn-mlp", 16, False, False), ("blocks+txt", 48, True, False), ("all", 56, True, True)],
)
def test_lora_targets_picks_which_linears_get_an_adapter(
    workspace: Workspace, targets: str, count: int, text_fusion: bool, io_layers: bool
) -> None:
    _, saved = trained_lora(workspace, lora={"targets": targets})

    modules = _modules(saved)
    assert len(modules) == count and _blocks(0, 1) <= modules
    assert any(m.startswith("txtfusion.layerwise_blocks") for m in modules) == text_fusion
    assert ("txtfusion.projector" in modules, modules >= IO_MODULES) == (io_layers, io_layers)


def test_an_unknown_target_preset_is_refused(workspace: Workspace) -> None:
    workspace.add_images(1)

    with pytest.raises(ValueError, match=r"lora\.targets must be one of .*'blocks\+txt', 'all'\), got 'mlp'"):
        run_lora(workspace.lora_config(lora={"targets": "mlp"}))


def test_rank_alpha_prefix_and_block_range_shape_the_adapters(workspace: Workspace) -> None:
    _, saved = trained_lora(workspace, lora={"rank": 8, "alpha": 2.5, "prefix": "unet.", "block_range": [1, 1]})

    assert {key.split(".")[0] for key in saved} == {"unet"}
    assert _modules({k.replace("unet.", "diffusion_model."): v for k, v in saved.items()}) == _blocks(1)
    assert saved["unet.blocks.1.attn.wq.lora_A.weight"].shape == (8, 64)
    assert saved["unet.blocks.1.attn.wq.alpha"].item() == 2.5


def test_the_alpha_defaults_to_the_rank_and_the_lora_dtype_is_independent_of_the_output(workspace: Workspace) -> None:
    config = workspace.lora_config(output={"dtype": "float32"})
    del config["lora"]["alpha"]
    workspace.add_images(1)

    trainer = run_lora(config)

    assert trainer.run_metadata["lora_dtype"] == "float32"
    assert {p.dtype for p in trainer.lora_network.trainable_parameters()} == {torch.float32}
    saved = trainer.lora_network.get_state_dict(None)
    assert {saved[k].item() for k in saved if k.endswith("alpha")} == {4.0}


def test_metadata_records_the_adapters_the_optimizer_the_shift_and_the_config(workspace: Workspace) -> None:
    trainer, _ = trained_lora(workspace, train={"ema_decay": 0.5})

    meta = read_metadata(workspace.out / "k2.safetensors")
    assert (meta["maltrainer_architecture"], meta["maltrainer_trainer"]) == ("krea2", "train_krea2_lora.py")
    assert (meta["ss_network_dim"], meta["ss_network_alpha"], meta["step"]) == ("4", "4", "final")
    assert (meta["maltrainer_lora_layers"], meta["maltrainer_lora_dtype"]) == ("16", "float32")
    assert meta["maltrainer_ema_decay"] == "0.5" and meta["maltrainer_dataset_samples"] == "2"
    assert json.loads(meta["maltrainer_optimizer_resolved"])["name"] == "automagic3"
    assert float(meta["maltrainer_resolved_sigma_shift"]) == pytest.approx(trainer.run_metadata["resolved_sigma_shift"])
    assert json.loads(meta["maltrainer_config"])["lora"] == {"rank": 4, "alpha": 4, "dtype": "float32"}


def test_with_ema_the_checkpoint_holds_the_average_and_the_raw_adapters_come_back(workspace: Workspace) -> None:
    trainer, saved = trained_lora(workspace, train={"ema_decay": 0.5, "ema_warmup": False})

    assert trainer.ema is not None
    network = trainer.accelerator.unwrap_model(trainer.lora_network)
    name = "diffusion_model.blocks.0.attn.wq.lora_B.weight"
    assert not torch.equal(network.get_state_dict(torch.bfloat16)[name], saved[name])
    with trainer.ema.applied():
        assert torch.equal(network.get_state_dict(torch.bfloat16)[name], saved[name])


@pytest.mark.parametrize("fp8", [False, True])
def test_the_base_loads_as_fp8_whether_the_file_stores_bf16_or_fp8(workspace: Workspace, fp8: bool) -> None:
    write_checkpoint(workspace.checkpoint, fp8=fp8)
    workspace.add_images(1)

    trainer = run_lora(workspace.lora_config(train={"max_steps": 1}))

    info = trainer.run_metadata["base_load"]
    assert isinstance(info, dict) and info["fp8_linears"] == 16
    assert ("quantized_on_load" in info, info["source_fp8_layers"] > 0) == (not fp8, fp8)
