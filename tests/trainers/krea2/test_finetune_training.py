"""A finetune on the tiny DiT: only the active windows and the text fusion change, saved as a full ComfyUI file."""

from __future__ import annotations

import logging

import pytest
import torch
from safetensors import safe_open
from torch import nn

from maltrainer.models.krea2.loading import load_krea2_transformer

from .runs import final_weights, run_finetune
from .tiny import tiny_model
from .world import PREFIX, Workspace, same_values, stored_weights, write_checkpoint

WINDOW_LEAVES = {
    "attention": ("attn.wq", "attn.wk", "attn.wv", "attn.gate", "attn.wo"),
    "mlp.gate": ("mlp.gate",),
    "mlp.up": ("mlp.up",),
    "mlp.down": ("mlp.down",),
}
TEXT_FUSION = {
    f"{name}.weight"
    for name, module in tiny_model().named_modules()
    if name.startswith("txtfusion.") and isinstance(module, nn.Linear)
}


def _changed(ws: Workspace) -> set[str]:
    before, after = stored_weights(ws.checkpoint), final_weights(ws.out)
    assert {PREFIX + key for key in before} == set(after)
    return {key for key in before if not torch.equal(before[key], after[PREFIX + key])}


@pytest.mark.parametrize(
    ("steps", "windows"),
    [
        (2, ["attention"]),
        (4, ["attention", "mlp.gate"]),
        (6, ["attention", "mlp.gate", "mlp.up"]),
        (8, list(WINDOW_LEAVES)),
    ],
)
def test_only_the_windows_that_trained_and_the_text_fusion_change_and_everything_else_is_bit_identical(
    workspace: Workspace, steps: int, windows: list[str]
) -> None:
    workspace.add_images(3)

    run_finetune(workspace.finetune_config(train={"max_steps": steps}))

    trained = {f"blocks.{i}.{leaf}.weight" for name in windows for leaf in WINDOW_LEAVES[name] for i in range(2)}
    assert _changed(workspace) == trained | TEXT_FUSION


def test_the_saved_file_is_a_full_bf16_model_with_the_sources_prefix_that_loads_back(workspace: Workspace) -> None:
    workspace.add_images(2)

    run_finetune(workspace.finetune_config())

    saved = final_weights(workspace.out)
    assert {tensor.dtype for tensor in saved.values()} == {torch.bfloat16}
    with safe_open(str(workspace.out / "ft.safetensors"), framework="pt") as handle:
        meta = handle.metadata()
    assert (meta["maltrainer_model_type"], meta["step"], meta["maltrainer_global_step"]) == (
        "krea2_finetune",
        "final",
        "3",
    )
    assert meta["maltrainer_windows"] == '["attention", "mlp.gate", "mlp.up", "mlp.down"]'
    reloaded, info = load_krea2_transformer(workspace.out / "ft.safetensors", torch.device("cpu"), torch.bfloat16)
    assert info["key_prefix"] == PREFIX and info["source_fp8_layers"] == 0
    assert reloaded.state_dict().keys() >= {key.removeprefix(PREFIX) for key in saved if key.endswith("wq.weight")}


def test_the_same_seed_gives_bit_identical_models(workspace: Workspace) -> None:
    workspace.add_images(3)
    train = {"max_steps": 4, "caption_dropout": 0.5, "timestep_distribution": "uniform"}

    run_finetune(workspace.finetune_config(train=train))
    run_finetune(workspace.finetune_config(train=train, output={"path": str(workspace.root / "again")}))

    assert same_values(final_weights(workspace.out), final_weights(workspace.root / "again"))


def test_an_fp8_source_file_trains_from_its_dequantized_weights_and_the_trainer_warns(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    write_checkpoint(workspace.checkpoint, fp8=True)
    workspace.add_images(1)

    with caplog.at_level(logging.WARNING):
        trainer = run_finetune(workspace.finetune_config(train={"max_steps": 1}))

    assert "krea2.safetensors stores its blocks in fp8: the finetune starts from dequantized fp8 weights" in caplog.text
    assert trainer.run_metadata["base_load"] == {"source_fp8_layers": 16, "fp8_linears": 16}
    assert {tensor.dtype for tensor in final_weights(workspace.out).values()} == {torch.bfloat16}
