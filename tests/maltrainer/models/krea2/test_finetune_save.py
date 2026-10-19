from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from maltrainer.finetune.rotation import Window, WindowRotator
from maltrainer.models.krea2 import loading
from maltrainer.models.krea2.finetune import DEFAULT_WINDOWS, save_krea2_model
from maltrainer.models.krea2.loading import is_block_linear, load_krea2_transformer
from tests.maltrainer.models.krea2.checkpoints import BLOCK_LINEARS, dense_state, write
from tests.maltrainer.models.krea2.tiny_model import TINY

CPU = torch.device("cpu")
SOURCE = dense_state(torch.bfloat16)


@pytest.fixture(autouse=True)
def tiny_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loading, "Krea2Config", lambda: TINY)


def _rotator(tmp_path: Path, windows: Sequence[Window] = DEFAULT_WINDOWS) -> WindowRotator:
    master: dict[str, torch.Tensor] = {}
    model, _ = load_krea2_transformer(write(tmp_path / "src.safetensors", SOURCE), CPU, torch.float32, master=master)
    return WindowRotator(model, master, windows, CPU, torch.float32, lambda params, _: torch.optim.SGD(params, lr=0.1))


def test_a_saved_file_holds_the_dense_layout_with_block_weights_from_the_master(tmp_path: Path) -> None:
    rotator = _rotator(tmp_path)
    path = tmp_path / "out.safetensors"

    save_krea2_model(rotator.model, rotator, path)

    saved = load_file(str(path))
    assert set(saved) == set(SOURCE)
    for key in SOURCE:
        if is_block_linear(key):
            assert saved[key].dtype == torch.bfloat16 and torch.equal(saved[key], rotator.master[key])
    assert torch.equal(saved["first.weight"], rotator.model.first.weight.detach().cpu())
    assert not any(key.endswith("weight_scale") for key in saved)


def test_the_file_is_written_aside_and_renamed_over_an_existing_one(tmp_path: Path) -> None:
    rotator = _rotator(tmp_path)
    path = tmp_path / "out.safetensors"
    path.write_bytes(b"stale")

    save_krea2_model(rotator.model, rotator, path)

    assert sorted(p.name for p in tmp_path.glob("out*")) == ["out.safetensors"]
    assert len(load_file(str(path))) == len(SOURCE)


def test_the_key_prefix_and_metadata_are_written_into_the_file(tmp_path: Path) -> None:
    rotator = _rotator(tmp_path)
    path = tmp_path / "out.safetensors"

    save_krea2_model(rotator.model, rotator, path, key_prefix="model.diffusion_model.", metadata={"step": "7"})

    with safe_open(str(path), framework="pt") as handle:
        assert all(key.startswith("model.diffusion_model.") for key in handle.keys())
        assert handle.metadata() == {"step": "7"}


def test_the_active_window_is_saved_live_and_the_others_from_the_master(tmp_path: Path) -> None:
    rotator = _rotator(tmp_path)
    rotator.activate(0)
    live = rotator.model.blocks[0].attn.wq.weight
    live.data += 1.0
    path = tmp_path / "out.safetensors"

    save_krea2_model(rotator.model, rotator, path)

    saved = load_file(str(path))
    assert torch.equal(saved["blocks.0.attn.wq.weight"], live.detach().to(torch.bfloat16))
    assert not torch.equal(saved["blocks.0.attn.wq.weight"], rotator.master["blocks.0.attn.wq.weight"])
    assert torch.equal(saved["blocks.0.mlp.up.weight"], SOURCE["blocks.0.mlp.up.weight"])


def test_block_weights_outside_every_window_are_still_saved_from_the_master(tmp_path: Path) -> None:
    rotator = _rotator(tmp_path, windows=(("attention", ("attn.",)),))
    path = tmp_path / "out.safetensors"

    save_krea2_model(rotator.model, rotator, path)

    saved = load_file(str(path))
    assert torch.equal(saved["blocks.1.mlp.gate.weight"], SOURCE["blocks.1.mlp.gate.weight"])
    assert "blocks.1.mlp.gate.weight_scale" not in saved


def test_a_saved_file_loads_back_with_the_same_master_and_prefix(tmp_path: Path) -> None:
    rotator = _rotator(tmp_path)
    rotator.activate(1)
    rotator.model.blocks[1].mlp.gate.weight.data *= 2.0
    path = tmp_path / "out.safetensors"
    save_krea2_model(rotator.model, rotator, path, key_prefix="diffusion_model.")

    master: dict = {}
    _, info = load_krea2_transformer(path, CPU, torch.float32, master=master)

    assert info["key_prefix"] == "diffusion_model." and info["quantized_on_load"] == BLOCK_LINEARS
    assert torch.equal(master["blocks.1.mlp.gate.weight"], SOURCE["blocks.1.mlp.gate.weight"] * 2.0)
