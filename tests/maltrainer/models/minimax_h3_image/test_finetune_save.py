"""The finetuned H3 Image file: the base file's layout with the trained weights in it, and the resume weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from maltrainer.finetune.rotation_int8 import Int8WindowRotator
from maltrainer.models.minimax_h3_image.finetune_save import read_resume_weights, write_int8_model, write_resume_weights
from tests.maltrainer.finetune.int8_model import GROUP, Model, rotator

MARKER = torch.tensor(
    list(json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": GROUP}).encode()),
    dtype=torch.uint8,
)
QKV0 = "blocks.0.attn.qkv_proj"


def _base_file(path: Path, model: Model) -> Path:
    tensors: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        tensors[f"{name}.weight"] = module.weight.detach().clone()
        if hasattr(module, "scale_weight"):
            tensors[f"{name}.weight_scale"] = module.scale_weight.detach().clone()
            tensors[f"{name}.comfy_quant"] = MARKER.clone()
    tensors["head.bias"] = model.head.bias.detach().clone()
    tensors["head.weight"] = model.head.weight.detach().to(torch.bfloat16)
    tensors["blocks.0.attn.comfy_attention.config"] = torch.tensor([1, 2, 3], dtype=torch.uint8)
    save_file(tensors, str(path), metadata={"config": '{"transformer": {"pdd_steps": 6}}'})
    return path


def _trained_rotator(model: Model) -> Int8WindowRotator:
    rot = rotator(model)
    rot.activate(0)
    with torch.no_grad():
        model.blocks[0].attn.qkv_proj.weight.add_(0.5)
    return rot


def test_the_saved_file_has_the_base_files_tensors_and_the_trained_codes(tmp_path: Path) -> None:
    model = Model()
    source = _base_file(tmp_path / "base.safetensors", model)
    rot = _trained_rotator(model)
    expected_codes, expected_scale = rot.int8(f"{QKV0}.weight")

    write_int8_model(source, rot, [], tmp_path / "out.safetensors", {"maltrainer_step": "7"})

    saved, base = load_file(str(tmp_path / "out.safetensors")), load_file(str(source))
    assert saved.keys() == base.keys()
    assert torch.equal(saved[f"{QKV0}.weight"], expected_codes) and torch.equal(
        saved[f"{QKV0}.weight_scale"], expected_scale
    )
    assert (
        saved[f"{QKV0}.weight"].dtype is torch.int8
        and saved[f"{QKV0}.weight_scale"].shape == base[f"{QKV0}.weight_scale"].shape
    )
    assert not torch.equal(saved[f"{QKV0}.weight"], base[f"{QKV0}.weight"])
    active = {f"{leaf}.{part}" for leaf in (QKV0, "blocks.0.mlp.fc1") for part in ("weight", "weight_scale")}
    for key in base.keys() - active:  # markers, hints, the frozen block: as they were
        assert torch.equal(saved[key], base[key])
    assert torch.equal(saved["blocks.0.mlp.fc1.weight"], base["blocks.0.mlp.fc1.weight"])  # untrained: same codes
    torch.testing.assert_close(saved["blocks.0.mlp.fc1.weight_scale"], base["blocks.0.mlp.fc1.weight_scale"])


def test_the_files_own_metadata_is_kept_and_wins_over_the_runs(tmp_path: Path) -> None:
    model = Model()
    source = _base_file(tmp_path / "base.safetensors", model)

    write_int8_model(source, rotator(model), [], tmp_path / "out.safetensors", {"config": "the run's", "extra": "1"})

    with safe_open(str(tmp_path / "out.safetensors"), framework="pt") as handle:
        assert handle.metadata() == {"config": '{"transformer": {"pdd_steps": 6}}', "extra": "1"}
    assert not list(tmp_path.glob("*.tmp"))


def test_an_always_trained_weight_is_saved_in_the_dtype_the_file_stores_it_in(tmp_path: Path) -> None:
    model = Model()
    source = _base_file(tmp_path / "base.safetensors", model)
    with torch.no_grad():
        model.head.weight.fill_(0.5)

    write_int8_model(source, rotator(model), [("head.weight", model.head.weight)], tmp_path / "out.safetensors")

    saved = load_file(str(tmp_path / "out.safetensors"))["head.weight"]
    assert saved.dtype is torch.bfloat16 and torch.equal(saved, torch.full((2, 4), 0.5, dtype=torch.bfloat16))


def test_resume_weights_hold_the_trained_blocks_and_the_always_trained_weights(tmp_path: Path) -> None:
    model = Model()
    rot = _trained_rotator(model)
    rot.activate(1)
    with torch.no_grad():
        model.head.weight.fill_(0.25)

    write_resume_weights(rot, [("head.weight", model.head.weight)], tmp_path / "w.safetensors")
    blocks, always = read_resume_weights(tmp_path / "w.safetensors")

    assert blocks.keys() == set(rot.keys()) and always.keys() == {"head.weight"}
    assert torch.equal(blocks[f"{QKV0}.weight"], rot.weight(f"{QKV0}.weight"))
    assert torch.equal(always["head.weight"], torch.full((2, 4), 0.25))


def test_a_save_that_fails_leaves_no_partial_file(tmp_path: Path) -> None:
    model = Model()
    source = _base_file(tmp_path / "base.safetensors", model)

    with pytest.raises(TypeError):
        write_int8_model(source, rotator(model), [], tmp_path / "out.safetensors", cast(Any, {"bad": 1}))

    assert not (tmp_path / "out.safetensors").exists()
