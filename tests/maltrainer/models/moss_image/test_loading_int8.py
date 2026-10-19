from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest
import torch
from safetensors.torch import load_file

from maltrainer.models.minimax_h3.int8_convrot import rotate_activation
from maltrainer.models.moss_image.loading import load_moss_image_transformer
from maltrainer.models.moss_image.lora import MOSSImageLoRANetwork
from maltrainer.models.moss_image.model import MOSSImageTransformer

from .tiny import CPU, GROUP_SIZE

Maker = Callable[..., MOSSImageTransformer]
Writer = Callable[..., Path]


def _inputs():
    torch.manual_seed(3)
    return torch.randn(1, 128, 2, 2), torch.full((1,), 0.4), torch.randn(1, 3, 8)


def test_an_int8_convrot_checkpoint_keeps_its_block_weights_int8_and_casts_the_rest(
    make_model: Maker, write_int8_checkpoint: Writer
) -> None:
    model, info = load_moss_image_transformer(write_int8_checkpoint(make_model()), CPU, torch.bfloat16)

    quantized = model.layers[1].proj["ffn_down"]
    assert info["format"] == "int8_convrot"
    assert info["int8_layers"] == 12
    assert quantized.weight.dtype == torch.int8
    assert quantized.scale_weight.dtype == torch.float32
    assert quantized.scale_weight.shape == (128, 1)
    assert quantized.int8_convrot_groupsize.item() == GROUP_SIZE
    assert model.video_embedder.weight.dtype == torch.bfloat16
    assert model.layers[0].norm1.weight.dtype == torch.bfloat16
    assert not any(p.requires_grad for p in model.parameters())


def test_a_quantized_linear_rotates_the_activation_then_applies_the_dequantized_weight(
    make_model: Maker, write_int8_checkpoint: Writer
) -> None:
    path = write_int8_checkpoint(make_model())
    stored = load_file(str(path))
    model, _ = load_moss_image_transformer(path, CPU, torch.float32)
    x = torch.randn(3, 128)

    out = model.layers[0].proj["o"](x)

    dense = stored["layers.0.proj.o.weight"].float() * stored["layers.0.proj.o.weight_scale"].float()[:, None]
    torch.testing.assert_close(out, rotate_activation(x, GROUP_SIZE) @ dense.T, atol=1e-4, rtol=1e-4)


def test_the_int8_model_approximates_the_float_model_it_was_quantized_from(
    make_model: Maker, write_int8_checkpoint: Writer
) -> None:
    source = make_model()
    model, _ = load_moss_image_transformer(write_int8_checkpoint(source), CPU, torch.float32)
    x, sigma, context = _inputs()

    with torch.no_grad():
        expected, got = source(x, sigma, context), model(x, sigma, context)

    assert (got - expected).norm() / expected.norm() < 0.05


@pytest.mark.parametrize(("requested", "stored"), [("fp32", "fp32"), ("bfloat16", "bf16"), ("bf16", "bf16")])
def test_the_gradient_precision_reaches_every_quantized_linear(
    make_model: Maker, write_int8_checkpoint: Writer, requested: str, stored: str
) -> None:
    model, _ = load_moss_image_transformer(
        write_int8_checkpoint(make_model()), CPU, torch.float32, grad_precision=requested
    )

    assert {m.int8_convrot_grad_precision for m in model.modules() if hasattr(m, "scale_weight")} == {stored}


def test_an_unknown_gradient_precision_is_rejected(make_model: Maker, write_int8_checkpoint: Writer) -> None:
    with pytest.raises(ValueError, match="Unknown INT8 ConvRot gradient precision"):
        load_moss_image_transformer(write_int8_checkpoint(make_model()), CPU, grad_precision="fp8")


def test_a_lora_on_the_int8_base_trains_through_the_frozen_quantized_layers(
    make_model: Maker, write_int8_checkpoint: Writer
) -> None:
    model, _ = load_moss_image_transformer(write_int8_checkpoint(make_model()), CPU, torch.float32)
    network = MOSSImageLoRANetwork(model, rank=2, targets="attn")
    for layer in network.lora_layers.values():
        layer.lora_up.weight.data.normal_(std=0.1)
    x, sigma, context = _inputs()

    network(x, sigma, context).square().sum().backward()

    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in network.trainable_parameters())
    assert all(p.grad is None for p in model.parameters())
