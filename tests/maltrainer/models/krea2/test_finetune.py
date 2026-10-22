"""What a Krea 2 finetune trains (the saved checkpoint is in test_finetune_save.py)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.finetune.rotation import WindowRotator
from maltrainer.models.krea2.finetune import (
    DEFAULT_ALWAYS_ON,
    DEFAULT_WINDOWS,
    always_on_linears,
    parse_windows,
)
from maltrainer.models.krea2.loading import load_krea2_transformer

from .tiny import CPU, tiny_model


def test_no_window_config_gives_the_four_default_windows():
    assert parse_windows(None) is DEFAULT_WINDOWS
    assert parse_windows([]) is DEFAULT_WINDOWS
    assert [name for name, _ in DEFAULT_WINDOWS] == ["attention", "mlp.gate", "mlp.up", "mlp.down"]
    assert DEFAULT_ALWAYS_ON == ("txtfusion",)


def test_windows_parse_from_dicts_pairs_and_single_string_prefixes():
    parsed = parse_windows(
        [
            {"name": "a", "prefixes": ["attn.", "mlp.gate"]},
            ["b", ["mlp.up"]],
            ("c", "mlp.down"),
            {"name": 4, "prefixes": "x"},
        ]
    )
    assert parsed == (("a", ("attn.", "mlp.gate")), ("b", ("mlp.up",)), ("c", ("mlp.down",)), ("4", ("x",)))


def test_always_on_linears_makes_text_fusion_weights_trainable_and_names_them():
    model = tiny_model().requires_grad_(False)
    found = always_on_linears(model, ["txtfusion"])
    names = [n for n, _ in found]
    assert len(found) == 33  # 4 blocks x (wq wk wv gate wo + mlp gate up down) + the layer projector
    assert "txtfusion.projector.weight" in names and "txtfusion.layerwise_blocks.0.attn.wq.weight" in names
    assert all(p.requires_grad for _, p in found)
    assert not model.blocks[0].attn.wq.weight.requires_grad  # only the named module


def test_always_on_linears_refuses_fp8_weights():
    root = nn.Module()
    root.enc = nn.Sequential(nn.Linear(2, 2, bias=False))
    root.enc[0].weight = nn.Parameter(torch.zeros(2, 2).to(torch.float8_e4m3fn), requires_grad=False)
    with pytest.raises(ValueError, match=r"enc\.0 is fp8; always-on modules must be bf16"):
        always_on_linears(root, ["enc"])


def test_always_on_linears_with_an_unknown_module_raises():
    with pytest.raises(AttributeError):
        always_on_linears(tiny_model(), ["nonexistent"])


@pytest.fixture
def loaded(write_checkpoint):
    master: dict = {}
    model, _ = load_krea2_transformer(
        write_checkpoint(prefix="model.diffusion_model."), CPU, compute_dtype=torch.float32, master=master
    )
    rotator = WindowRotator(
        model, master, DEFAULT_WINDOWS, CPU, torch.float32, lambda params, _name: torch.optim.SGD(params, lr=0.1)
    )
    return model, rotator


def test_default_windows_split_the_tiny_blocks_linears(loaded):
    _, rotator = loaded
    sizes = {name: len(keys) for name, keys in rotator.windows}
    assert sizes == {"attention": 10, "mlp.gate": 2, "mlp.up": 2, "mlp.down": 2}
