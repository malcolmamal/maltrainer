"""Loading a Krea 2 DiT from a ComfyUI single file: bf16 (quantized on load) or fp8_scaled."""

from __future__ import annotations

import pytest
import torch

from maltrainer.models.comfy_fp8 import Fp8ScaledLinear, dequantize
from maltrainer.models.krea2.loading import is_block_linear, load_krea2_transformer

from .tiny import CPU, inputs, tiny_model

BLOCK_LINEARS = 16  # 2 blocks x (wq wk wv gate wo + mlp gate up down)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("blocks.0.attn.wq.weight", True),
        ("blocks.12.attn.gate.weight", True),
        ("blocks.3.mlp.down.weight", True),
        ("blocks.0.attn.wq.bias", False),
        ("blocks.0.attn.qknorm.qnorm.scale", False),
        ("blocks.0.mod.lin", False),
        ("txtfusion.layerwise_blocks.0.attn.wq.weight", False),
        ("txtfusion.refiner_blocks.0.mlp.up.weight", False),
        ("first.weight", False),
    ],
)
def test_is_block_linear_picks_exactly_the_dit_block_attention_and_mlp_weights(key, expected):
    assert is_block_linear(key) is expected


def test_bf16_file_is_quantized_per_row_to_fp8_on_load(write_checkpoint):
    model, info = load_krea2_transformer(write_checkpoint(), CPU)
    assert info == {
        "source_fp8_layers": 0,
        "key_prefix": "",
        "quantized_on_load": BLOCK_LINEARS,
        "fp8_linears": BLOCK_LINEARS,
    }
    original = tiny_model()
    linear = model.blocks[1].mlp.up
    assert isinstance(linear, Fp8ScaledLinear) and linear.weight.dtype == torch.float8_e4m3fn
    assert linear.weight_scale.shape == (linear.out_features, 1)
    rebuilt = dequantize(linear.weight, linear.weight_scale, torch.float32)
    reference = original.blocks[1].mlp.up.weight.to(torch.bfloat16).float()
    row_max = reference.abs().amax(dim=1, keepdim=True)
    assert ((rebuilt - reference).abs() <= 0.07 * row_max).all()  # e4m3 keeps ~3 mantissa bits


def test_layers_outside_the_blocks_stay_plain_linears_in_the_compute_dtype(write_checkpoint):
    model, _ = load_krea2_transformer(write_checkpoint(), CPU, compute_dtype=torch.float32)
    assert type(model.first) is torch.nn.Linear and model.first.weight.dtype == torch.float32
    assert type(model.txtfusion.refiner_blocks[0].attn.wq) is torch.nn.Linear
    assert model.compute_dtype == torch.float32
    assert not any(p.requires_grad for p in model.parameters())
    assert torch.allclose(model.first.weight, tiny_model().first.weight.to(torch.bfloat16).float())


def test_loaded_model_runs_and_stays_close_to_the_bf16_original(write_checkpoint):
    model, _ = load_krea2_transformer(write_checkpoint(), CPU, compute_dtype=torch.float32)
    x, sigma, context = inputs(batch=1)
    out = model(x, sigma, context)
    reference = tiny_model()(x, sigma, context)
    assert out.shape == x.shape
    assert torch.nn.functional.cosine_similarity(out.flatten(), reference.flatten(), dim=0) > 0.98


def test_master_gets_the_files_own_bf16_block_weights(write_checkpoint):
    master: dict = {}
    load_krea2_transformer(write_checkpoint(), CPU, master=master)
    assert len(master) == BLOCK_LINEARS and all(is_block_linear(k) for k in master)
    expected = tiny_model().blocks[0].attn.wk.weight.to(torch.bfloat16)
    assert master["blocks.0.attn.wk.weight"].dtype == torch.bfloat16
    assert torch.equal(master["blocks.0.attn.wk.weight"], expected)


def test_container_prefix_is_reported_and_stripped(write_checkpoint):
    model, info = load_krea2_transformer(write_checkpoint(prefix="model.diffusion_model."), CPU)
    assert info["key_prefix"] == "model.diffusion_model."
    assert info["fp8_linears"] == BLOCK_LINEARS
    assert isinstance(model.blocks[0].attn.wq, Fp8ScaledLinear)


def test_fp8_scaled_file_keeps_its_exact_fp8_weights(write_checkpoint):
    path = write_checkpoint(prefix="diffusion_model.", fp8=True)
    master: dict = {}
    model, info = load_krea2_transformer(path, CPU, compute_dtype=torch.float32, master=master)
    assert info["source_fp8_layers"] == BLOCK_LINEARS and "quantized_on_load" not in info
    assert info["key_prefix"] == "diffusion_model."
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        stored = handle.get_tensor("diffusion_model.blocks.0.attn.wq.weight")
        scale = handle.get_tensor("diffusion_model.blocks.0.attn.wq.weight_scale")
    layer = model.blocks[0].attn.wq
    assert torch.equal(layer.weight.view(torch.uint8), stored.view(torch.uint8))
    assert torch.equal(layer.weight_scale, scale)
    assert torch.equal(master["blocks.0.attn.wq.weight"], dequantize(stored, scale, torch.bfloat16))


def test_file_missing_a_weight_is_refused_naming_the_key(write_checkpoint):
    path = write_checkpoint(drop=("last.linear.weight",))
    with pytest.raises(ValueError, match=r"does not match the Krea 2 DiT.*last\.linear\.weight"):
        load_krea2_transformer(path, CPU)


def test_file_with_an_unknown_tensor_is_refused(write_checkpoint):
    path = write_checkpoint(extra={"blocks.0.surprise": torch.zeros(2)})
    with pytest.raises(ValueError, match=r"unexpected \['blocks\.0\.surprise'\]"):
        load_krea2_transformer(path, CPU)
