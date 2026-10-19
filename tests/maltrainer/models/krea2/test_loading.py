from __future__ import annotations

from pathlib import Path

import pytest
import torch

from maltrainer.models.comfy_fp8 import Fp8ScaledLinear, dequantize, quantize_rows
from maltrainer.models.krea2 import loading
from maltrainer.models.krea2.loading import is_block_linear, load_krea2_transformer
from tests.maltrainer.models.krea2.checkpoints import BLOCK_LINEARS, dense_state, fp8_layer_names, fp8_state, write
from tests.maltrainer.models.krea2.tiny_model import TINY, tiny_inputs, tiny_transformer

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def tiny_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loading, "Krea2Config", lambda: TINY)


def _fp8_file(tmp_path: Path, prefix: str = "") -> Path:
    state = fp8_state()
    return write(tmp_path / "krea2_fp8.safetensors", state, prefix, fp8_layers=fp8_layer_names(state))


def _dequantized_reference(state: dict) -> torch.nn.Module:
    reference = tiny_transformer()
    dense = {k: v.float() for k, v in state.items() if not k.endswith("weight_scale")}
    for key in [k for k in state if is_block_linear(k)]:
        dense[key] = dequantize(state[key], state[key.replace(".weight", ".weight_scale")], torch.float32)
    reference.load_state_dict(dense)
    return reference


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("blocks.0.attn.wq.weight", True),
        ("blocks.3.attn.gate.weight", True),
        ("blocks.1.mlp.up.weight", True),
        ("blocks.0.attn.wq.weight_scale", False),
        ("blocks.0.attn.qknorm.qnorm.scale", False),
        ("blocks.0.mod.lin", False),
        ("txtfusion.layerwise_blocks.0.attn.wq.weight", False),
        ("first.weight", False),
    ],
)
def test_only_the_block_attention_and_mlp_weights_are_quantized(key: str, expected: bool) -> None:
    assert is_block_linear(key) is expected


def test_a_bf16_file_is_quantized_to_fp8_per_output_row_on_load(tmp_path: Path) -> None:
    path = write(tmp_path / "bf16.safetensors", dense_state(torch.bfloat16))

    model, info = load_krea2_transformer(path, CPU, compute_dtype=torch.float32)

    wq = model.blocks[0].attn.wq
    assert isinstance(wq, Fp8ScaledLinear)
    assert wq.weight.dtype == torch.float8_e4m3fn and wq.weight_scale.shape == (32, 1)
    assert info == {
        "source_fp8_layers": 0,
        "key_prefix": "",
        "quantized_on_load": BLOCK_LINEARS,
        "fp8_linears": BLOCK_LINEARS,
    }
    assert not isinstance(model.txtfusion.layerwise_blocks[0].attn.wq, Fp8ScaledLinear)


def test_a_quantized_bf16_model_computes_with_the_row_quantized_weights(tmp_path: Path) -> None:
    state = dense_state(torch.bfloat16)
    inputs = tiny_inputs()

    model, _ = load_krea2_transformer(write(tmp_path / "bf16.safetensors", state), CPU, compute_dtype=torch.float32)

    reference = tiny_transformer()
    dense = {k: v.float() for k, v in state.items()}
    for key in [k for k in state if is_block_linear(k)]:
        dense[key] = dequantize(*quantize_rows(state[key]), torch.float32)
    reference.load_state_dict(dense)
    torch.testing.assert_close(model(*inputs), reference(*inputs), atol=1e-3, rtol=1e-3)


def test_an_fp8_scaled_file_keeps_its_stored_weights_and_scales_under_the_container_prefix(tmp_path: Path) -> None:
    state = fp8_state()
    path = _fp8_file(tmp_path, prefix="model.diffusion_model.")

    model, info = load_krea2_transformer(path, CPU, compute_dtype=torch.float32)

    wq = model.blocks[1].attn.wq
    assert torch.equal(wq.weight.float(), state["blocks.1.attn.wq.weight"].float())
    assert wq.weight_scale.item() == state["blocks.1.attn.wq.weight_scale"].item()
    assert info == {"source_fp8_layers": BLOCK_LINEARS, "key_prefix": "model.diffusion_model.", "fp8_linears": 16}


def test_an_fp8_scaled_model_computes_with_the_dequantized_weights(tmp_path: Path) -> None:
    inputs = tiny_inputs()

    model, _ = load_krea2_transformer(_fp8_file(tmp_path), CPU, compute_dtype=torch.float32)

    reference = _dequantized_reference(fp8_state())
    torch.testing.assert_close(model(*inputs), reference(*inputs), atol=1e-3, rtol=1e-3)


def test_the_model_is_frozen_and_the_other_tensors_take_the_compute_dtype(tmp_path: Path) -> None:
    model, _ = load_krea2_transformer(_fp8_file(tmp_path), CPU, compute_dtype=torch.bfloat16)

    assert model.compute_dtype == torch.bfloat16
    assert not any(p.requires_grad for p in model.parameters())
    assert model.first.weight.dtype == torch.bfloat16
    assert model.blocks[0].mod.lin.dtype == torch.bfloat16
    assert model.blocks[0].attn.wq.weight_scale.dtype == torch.float32
    assert model.blocks[0].attn.wq.weight.dtype == torch.float8_e4m3fn


def test_the_master_copy_of_a_bf16_file_is_the_files_own_block_weights(tmp_path: Path) -> None:
    state = dense_state(torch.bfloat16)
    master: dict = {}

    load_krea2_transformer(write(tmp_path / "bf16.safetensors", state), CPU, master=master)

    assert len(master) == BLOCK_LINEARS
    assert all(torch.equal(master[k], state[k]) and master[k].dtype == torch.bfloat16 for k in master)


def test_the_master_copy_of_an_fp8_file_is_dequantized_to_bf16(tmp_path: Path) -> None:
    state = fp8_state()
    master: dict = {}

    load_krea2_transformer(_fp8_file(tmp_path), CPU, master=master)

    key = "blocks.0.mlp.down.weight"
    expected = dequantize(state[key], state[key.replace(".weight", ".weight_scale")], torch.bfloat16)
    assert torch.equal(master[key], expected)
    assert set(master) == {k for k in state if is_block_linear(k)}


@pytest.mark.parametrize("damage", ["missing", "unexpected"])
def test_a_file_that_is_not_the_krea2_dit_is_refused(tmp_path: Path, damage: str) -> None:
    state = dense_state(torch.bfloat16)
    if damage == "missing":
        del state["first.bias"]
    else:
        state["blocks.0.bogus"] = torch.zeros(1)

    with pytest.raises(ValueError, match="does not match the Krea 2 DiT"):
        load_krea2_transformer(write(tmp_path / "bad.safetensors", state), CPU)
