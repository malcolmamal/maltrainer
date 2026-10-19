"""What a Krea 2 finetune trains, and the checkpoint it saves."""

from __future__ import annotations

import pytest
import torch
from safetensors import safe_open
from torch import nn

from maltrainer.finetune.rotation import WindowRotator
from maltrainer.models.krea2.finetune import (
    DEFAULT_ALWAYS_ON,
    DEFAULT_WINDOWS,
    always_on_linears,
    parse_windows,
    save_krea2_model,
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


def test_saved_file_has_the_source_prefix_and_every_model_key(loaded, tmp_path):
    model, rotator = loaded
    path = tmp_path / "out.safetensors"
    save_krea2_model(model, rotator, path, key_prefix="model.diffusion_model.", metadata={"note": "x"})
    with safe_open(str(path), framework="pt") as handle:
        keys = set(handle.keys())
        assert handle.metadata()["note"] == "x"
    expected = {"model.diffusion_model." + k for k in tiny_model().state_dict()}
    assert keys == expected
    assert not (tmp_path / "out.safetensors.tmp").exists()


def test_save_writes_the_master_for_frozen_windows_and_the_live_weights_for_the_active_one(loaded, tmp_path):
    model, rotator = loaded
    rotator.activate(1)  # mlp.gate trains
    with torch.no_grad():
        for key in rotator.windows[1][1]:
            model.get_submodule(key[: -len(".weight")]).weight += 1.0
    path = tmp_path / "out.safetensors"
    save_krea2_model(model, rotator, path)
    original = tiny_model().state_dict()
    with safe_open(str(path), framework="pt") as handle:
        live = handle.get_tensor("blocks.0.mlp.gate.weight")
        frozen = handle.get_tensor("blocks.0.attn.wq.weight")
        io = handle.get_tensor("first.weight")
        assert live.dtype == frozen.dtype == torch.bfloat16
        assert "blocks.0.attn.wq.weight_scale" not in set(handle.keys())
    assert torch.equal(live, (original["blocks.0.mlp.gate.weight"].bfloat16().float() + 1.0).bfloat16())
    assert torch.equal(frozen, original["blocks.0.attn.wq.weight"].bfloat16())  # the master, not the fp8 copy
    assert torch.equal(io, original["first.weight"].bfloat16().float())  # the model's own tensor, dtype as held


def test_saved_file_loads_back_through_the_krea2_loader(loaded, tmp_path):
    model, rotator = loaded
    path = tmp_path / "round.safetensors"
    save_krea2_model(model, rotator, path, key_prefix="diffusion_model.")
    again, info = load_krea2_transformer(path, CPU, compute_dtype=torch.float32)
    assert info["key_prefix"] == "diffusion_model." and info["quantized_on_load"] == 16
    assert again.blocks[1].mlp.down.weight.shape == model.blocks[1].mlp.down.weight.shape
