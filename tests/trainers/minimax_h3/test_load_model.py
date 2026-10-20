"""Loading the base transformer from a checkpoint's tensors and injecting the adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import train_minimax_h3_lora as h3
from tests.maltrainer.models.minimax_h3.h3_adapters import hand_pair, write_adapter

from .fakes import Fakes
from .runs import cached, linear, lora_layer
from .world import Workspace, same_tensors


def _loaded(ws: Workspace, **sections: Any) -> h3.MiniMaxH3LoRATrainer:
    ws.add_images(1)
    trainer = cached(ws.config(**sections))
    trainer.load_transformer_and_lora()
    return trainer


def _layers(trainer: h3.MiniMaxH3LoRATrainer) -> list[str]:
    return sorted(trainer.lora_network.name_map.values())


def test_the_transformer_is_the_checkpoints_tensors_frozen_with_checkpointing_on(
    workspace: Workspace, fakes: Fakes
) -> None:
    trainer = _loaded(workspace)

    assert same_tensors(dict(trainer.transformer.state_dict()), fakes.reference_state())
    assert not any(p.requires_grad for p in trainer.transformer.parameters())
    assert trainer.transformer.gradient_checkpointing and fakes.loads["transformer"] == 1
    assert fakes.configs[-1].adaln_t_table_size is None


def test_a_pruned_checkpoint_gets_its_table_size_and_embedding_width_from_the_table(
    workspace: Workspace, fakes: Fakes
) -> None:
    fakes.adaln_table = 4

    trainer = _loaded(workspace)

    assert (fakes.configs[-1].adaln_t_table_size, fakes.configs[-1].time_embed_dim) == (4, 10)
    assert trainer.transformer.time_embedder is None and trainer.transformer.adaln_t_table.shape == (4, 10)


def test_missing_checkpoint_tensors_are_refused_by_name(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = fakes.reference_state()
    del state["blocks.0.mlp.fc1.weight"], state["rope.inv_freq"]
    monkeypatch.setattr(fakes, "reference_state", lambda: dict(state))

    with pytest.raises(
        ValueError, match=r"missing 2 tensor\(s\); examples: \['rope.inv_freq', 'blocks.0.mlp.fc1.weight'"
    ):
        _loaded(workspace)


def test_tensors_the_model_does_not_expect_are_refused_by_name(
    workspace: Workspace, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {**fakes.reference_state(), "extra.weight": torch.zeros(1)}
    monkeypatch.setattr(fakes, "reference_state", lambda: dict(state))

    with pytest.raises(ValueError, match=r"has 1 tensor\(s\) the model does not expect; examples: \['extra.weight'\]"):
        _loaded(workspace)


@pytest.mark.parametrize(("given", "recorded"), [(None, "bf16"), ("fp32", "fp32"), ("int8", "int8")])
def test_the_gradient_precision_is_recorded_as_resolved(workspace: Workspace, given: str | None, recorded: str) -> None:
    trainer = _loaded(workspace, train={"int8_grad_precision": given})

    assert trainer.run_metadata["int8_grad_precision"] == recorded


def test_an_unknown_gradient_precision_is_refused(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match=r"Unknown INT8 ConvRot gradient precision 'int4'"):
        _loaded(workspace, train={"int8_grad_precision": "int4"})


def test_by_default_the_four_projections_of_every_block_and_the_token_refiner_get_adapters(
    workspace: Workspace,
) -> None:
    trainer = _loaded(workspace)

    assert len(_layers(trainer)) == 16 and trainer.lora_network.prefix == "diffusion_model."
    assert trainer.run_metadata["lora_layers"] == 16 and trainer.run_metadata["trainable_parameters"] == 4 * 608
    assert trainer.run_metadata["lora_dtype"] == "float32"
    assert {p.dtype for p in trainer.lora_network.trainable_parameters()} == {torch.float32}


@pytest.mark.parametrize(
    ("lora", "expected"),
    [
        ({"block_range": [1, 2], "include_token_refiner": False}, 8),
        ({"block_range": [0, 0]}, 8),
        ({"include_token_refiner": False}, 12),
        ({"target_modules": ["fc1"]}, 4),
        ({"include_adaln": True}, 19),
        ({"include_adaln": True, "block_range": [2, 2], "include_token_refiner": False}, 5),
    ],
)
def test_block_range_token_refiner_targets_and_adaln_decide_which_layers_get_adapters(
    workspace: Workspace, lora: dict[str, Any], expected: int
) -> None:
    assert len(_layers(_loaded(workspace, lora=lora))) == expected


def test_adaln_adapters_cover_the_per_block_projections_only(workspace: Workspace) -> None:
    layers = _layers(_loaded(workspace, lora={"include_adaln": True, "target_modules": ["fc2"]}))

    assert [name for name in layers if "adaln" in name] == [f"blocks.{i}.adaln_proj.linear" for i in range(3)]


def test_targets_matching_nothing_are_refused(workspace: Workspace) -> None:
    with pytest.raises(ValueError, match="No MiniMax H3 modules matched the LoRA targets / block_range"):
        _loaded(workspace, lora={"target_modules": ["nothing"]})


def test_rank_alpha_prefix_and_dtype_shape_the_adapters(workspace: Workspace) -> None:
    trainer = _loaded(workspace, lora={"rank": 2, "alpha": 8.0, "prefix": "unet.", "dtype": "bfloat16"})

    layer = lora_layer(trainer, "blocks_0_attn_qkv_proj")
    assert (layer.lora_down.weight.shape, layer.lora_up.weight.shape) == ((2, 16), (48, 2))
    assert layer.alpha == 8.0 and trainer.lora_network.prefix == "unet."
    assert {p.dtype for p in trainer.lora_network.trainable_parameters()} == {torch.bfloat16}
    assert linear(trainer, "blocks.0.attn.qkv_proj").weight.dtype == torch.float32
    assert trainer.run_metadata["lora_dtype"] == "bfloat16"


def _assistant(ws: Workspace) -> Path:
    down, up = hand_pair(16, 16)
    return write_adapter(ws.root / "assistant.safetensors", {"blocks.0.mlp.fc1": (down, up, 2.0)})
