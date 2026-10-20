"""A LoRA run on the tiny DiT: only the adapters train, and the saved file is the LoRA ComfyUI loads."""

from __future__ import annotations

import json

import pytest
import torch

from maltrainer.utils.lora_metadata import read_metadata

from .runs import run_lora, trained
from .world import Workspace, base_weights, named_weights, same_values

BLOCK_LEAVES = ("q_proj", "k_proj", "v_proj", "attn_out", "mlp_in", "mlp_out")
IO_MODULES = {
    "emb_in.image",
    "txt_in",
    "time_in.in_layer",
    "time_in.out_layer",
    "early_stream_modulations.image.lin",
    "early_stream_modulations.txt.lin",
    "single_stream_modulations.image.lin",
    "single_stream_modulations.txt.lin",
    "final_layer.image.linear",
    "final_layer.image.adaLN_modulation.1",
}


def _modules(saved: dict[str, torch.Tensor]) -> set[str]:
    return {key.removeprefix("diffusion_model.").rsplit(".", 2)[0] for key in saved if key.endswith("lora_A.weight")}


def _blocks(*sections: str) -> set[str]:
    return {
        f"{section}{i}.{leaf}"
        for section, count in (("txt_mode_blocks.", 1), ("content_mode_blocks.image.", 1), ("single_blocks.", 2))
        if section in sections
        for i in range(count)
        for leaf in BLOCK_LEAVES
    }


def test_only_the_adapters_train_and_the_base_stays_bit_identical(workspace: Workspace) -> None:
    trainer, saved = trained(workspace)

    assert same_values(named_weights(trainer.transformer), base_weights(workspace))
    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    assert all(torch.count_nonzero(saved[key]) > 0 for key in saved if key.endswith("lora_B.weight"))


def test_the_saved_file_holds_an_a_b_and_alpha_per_adapter_in_the_output_dtype(workspace: Workspace) -> None:
    _, saved = trained(workspace)

    assert len(saved) == 3 * 24
    assert saved["diffusion_model.single_blocks.0.q_proj.lora_A.weight"].shape == (4, 128)
    assert saved["diffusion_model.single_blocks.0.mlp_out.lora_B.weight"].shape == (128, 4)
    assert {saved[k].dtype for k in saved if not k.endswith("alpha")} == {torch.bfloat16}
    assert {saved[k].item() for k in saved if k.endswith("alpha")} == {4.0}


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        ("blocks", _blocks("txt_mode_blocks.", "content_mode_blocks.image.", "single_blocks.")),
        ("single", _blocks("single_blocks.")),
        ("all", _blocks("txt_mode_blocks.", "content_mode_blocks.image.", "single_blocks.") | IO_MODULES),
    ],
)
def test_lora_targets_picks_which_linears_get_an_adapter(
    workspace: Workspace, targets: str, expected: set[str]
) -> None:
    _, saved = trained(workspace, lora={"targets": targets})

    assert _modules(saved) == expected


def test_an_unknown_target_preset_is_refused(workspace: Workspace) -> None:
    workspace.add_images(1)

    with pytest.raises(ValueError, match=r"lora\.targets must be one of \('blocks', 'single', 'all'\), got 'mlp'"):
        run_lora(workspace.config(lora={"targets": "mlp"}))


def test_rank_alpha_and_prefix_shape_the_adapters(workspace: Workspace) -> None:
    _, saved = trained(workspace, lora={"rank": 8, "alpha": 2.5, "prefix": "unet."})

    assert {key.split(".")[0] for key in saved} == {"unet"}
    assert saved["unet.single_blocks.1.k_proj.lora_A.weight"].shape == (8, 128)
    assert saved["unet.single_blocks.1.k_proj.alpha"].item() == 2.5


def test_the_alpha_defaults_to_the_rank_and_the_lora_dtype_is_independent_of_the_output(workspace: Workspace) -> None:
    config = workspace.config(output={"dtype": "float32"})
    del config["lora"]["alpha"]
    workspace.add_images(1)

    trainer = run_lora(config)

    assert trainer.run_metadata["lora_dtype"] == "float32"
    assert {p.dtype for p in trainer.lora_network.trainable_parameters()} == {torch.float32}
    saved = trainer.lora_network.get_state_dict(None)
    assert {saved[k].item() for k in saved if k.endswith("alpha")} == {4.0}


def test_metadata_records_the_adapters_the_optimizer_and_the_config(workspace: Workspace) -> None:
    trainer, _ = trained(workspace, train={"ema_decay": 0.5})

    meta = read_metadata(workspace.out / "klein.safetensors")
    assert (meta["maltrainer_architecture"], meta["maltrainer_trainer"]) == ("flux3_klein", "train_flux3_klein_lora.py")
    assert (meta["ss_network_dim"], meta["ss_network_alpha"], meta["step"]) == ("4", "4", "final")
    assert (meta["maltrainer_lora_layers"], meta["maltrainer_lora_dtype"]) == ("24", "float32")
    assert (meta["maltrainer_int8_grad_precision"], meta["maltrainer_ema_decay"]) == ("bf16", "0.5")
    assert json.loads(meta["maltrainer_optimizer_resolved"])["name"] == "automagic3"
    assert int(meta["maltrainer_trainable_parameters"]) == trainer.run_metadata["trainable_parameters"]
    assert json.loads(meta["maltrainer_config"])["lora"] == {"rank": 4, "alpha": 4, "dtype": "float32"}


def test_with_ema_the_checkpoint_holds_the_average_and_the_raw_adapters_come_back(workspace: Workspace) -> None:
    trainer, saved = trained(workspace, train={"ema_decay": 0.5, "ema_warmup": False})

    assert trainer.ema is not None
    network = trainer.accelerator.unwrap_model(trainer.lora_network)
    name = "diffusion_model.single_blocks.0.q_proj.lora_B.weight"
    assert not torch.equal(network.get_state_dict(torch.bfloat16)[name], saved[name])
    with trainer.ema.applied():
        assert torch.equal(network.get_state_dict(torch.bfloat16)[name], saved[name])
