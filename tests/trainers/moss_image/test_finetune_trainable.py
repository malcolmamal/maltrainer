"""A finetune step moves the trainable parameters and leaves every other one bit-identical."""

from __future__ import annotations

import logging

import pytest

from .runs import changed, prepared_finetune, run_finetune
from .world import Workspace, base_weights, named_weights

BLOCK_LINEARS = {
    f"layers.{i}.proj.{leaf}.weight" for i in range(2) for leaf in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
}


def _trained(ws: Workspace, **finetune_options: object) -> tuple[set[str], set[str]]:
    ws.add_images("p", 2)
    trainer = run_finetune(ws.config(finetune=finetune_options, train={"max_steps": 2}))
    moved = changed(base_weights(ws), named_weights(trainer.transformer))
    return moved, {name for name, _ in trainer.named_trainable}


def test_the_default_trains_the_linear_weights_of_the_blocks_only(workspace: Workspace) -> None:
    moved, trainable = _trained(workspace)

    assert trainable == BLOCK_LINEARS
    assert moved == BLOCK_LINEARS


def test_a_prefix_narrows_training_to_the_blocks_it_names(workspace: Workspace) -> None:
    moved, trainable = _trained(workspace, trainable_prefixes=["layers.1."])

    assert trainable == {name for name in BLOCK_LINEARS if name.startswith("layers.1.")}
    assert moved == trainable


def test_an_empty_prefix_list_trains_everything_and_only_trained_parameters_move(workspace: Workspace) -> None:
    moved, trainable = _trained(workspace, trainable_prefixes=[], linears_only=False)

    assert trainable == set(base_weights(workspace))
    assert BLOCK_LINEARS | {"text_embedder.proj.weight", "video_embedder.weight", "video_embedder.bias"} <= moved
    assert moved <= trainable


def test_training_the_text_embedder_is_allowed_with_a_warning(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        moved, _ = _trained(workspace, trainable_prefixes=["text_embedder."])

    assert "text_embedder.proj.weight" in moved
    assert moved.isdisjoint(BLOCK_LINEARS)
    assert "trains the text adapter" in caplog.text


def test_linears_only_leaves_biases_and_norm_scales_frozen_inside_a_selected_prefix(workspace: Workspace) -> None:
    _, trainable = _trained(workspace, trainable_prefixes=["text_embedder."], linears_only=True)

    assert trainable == {
        "text_embedder.proj.weight",
        *(
            f"text_embedder.blocks.{i}.{leaf}.weight"
            for i in range(2)
            for leaf in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")
        ),
    }


def test_trainable_parameters_require_grad_and_the_rest_do_not(workspace: Workspace) -> None:
    workspace.add_images("p", 1)

    trainer = prepared_finetune(workspace.config())

    flags = {name for name, parameter in trainer.transformer.named_parameters() if parameter.requires_grad}
    assert flags == BLOCK_LINEARS
    assert trainer.transformer.training
