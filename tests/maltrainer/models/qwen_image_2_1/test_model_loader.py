from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from maltrainer.models.minimax_h3.int8_convrot import rotate_activation
from maltrainer.models.qwen_image_2_1 import model as model_module
from maltrainer.models.qwen_image_2_1.model import QwenImage21Transformer, load_qwen_image_2_1_transformer

from ._tiny import int8_checkpoint, tiny_inputs, tiny_transformer

CPU = torch.device("cpu")
QUANTIZED = {"img_in": 4, "transformer_blocks.0.attn.to_q": 16}


@pytest.fixture(autouse=True)
def tiny_architecture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "QwenImage21Transformer", tiny_transformer)


def _checkpoint(tmp_path: Path, quantized: dict[str, int] | None = None) -> tuple[Path, QwenImage21Transformer, dict]:
    source = tiny_transformer(seed=7)
    path = tmp_path / "transformer.safetensors"
    stored = int8_checkpoint(path, source.state_dict(), QUANTIZED if quantized is None else quantized)
    return path, source, stored


def test_a_missing_checkpoint_is_reported_with_its_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Transformer model file not found"):
        load_qwen_image_2_1_transformer(tmp_path / "absent.safetensors", device=CPU)


def test_the_loader_accepts_a_string_path_and_returns_a_frozen_model_in_eval_mode(tmp_path: Path) -> None:
    path, _, _ = _checkpoint(tmp_path)

    model = load_qwen_image_2_1_transformer(str(path), device=CPU, dtype=torch.float32)

    assert isinstance(model, QwenImage21Transformer)
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training


def test_quantized_linears_keep_their_int8_weights_while_the_rest_is_cast_to_the_dtype(tmp_path: Path) -> None:
    path, source, stored = _checkpoint(tmp_path)

    model = load_qwen_image_2_1_transformer(path, device=CPU, dtype=torch.bfloat16)

    assert model.img_in.weight.dtype == torch.int8
    assert torch.equal(model.img_in.weight, stored["img_in"])
    assert model.transformer_blocks[0].attn.to_q.int8_convrot_groupsize.item() == 16
    assert model.proj_out.weight.dtype == torch.bfloat16
    torch.testing.assert_close(model.proj_out.weight, source.proj_out.weight.to(torch.bfloat16))
    assert not hasattr(model.proj_out, "scale_weight")


def test_a_quantized_linear_computes_with_the_rotated_input_and_the_dequantized_weight(tmp_path: Path) -> None:
    path, _, stored = _checkpoint(tmp_path)
    model = load_qwen_image_2_1_transformer(path, device=CPU, dtype=torch.float32)
    x = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(3))

    out = model.img_in(x)

    dense = stored["img_in"].float() * model.img_in.scale_weight
    torch.testing.assert_close(out, rotate_activation(x, 4) @ dense.T, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(("requested", "expected"), [("bf16", "bf16"), ("fp32", "fp32"), ("float32", "fp32")])
def test_the_gradient_precision_is_applied_to_every_quantized_linear(
    tmp_path: Path, requested: str, expected: str
) -> None:
    path, _, _ = _checkpoint(tmp_path)

    model = load_qwen_image_2_1_transformer(path, device=CPU, dtype=torch.float32, grad_precision=requested)

    assert model.img_in.int8_convrot_grad_precision == expected
    assert model.transformer_blocks[0].attn.to_q.int8_convrot_grad_precision == expected


def test_an_unknown_gradient_precision_is_rejected(tmp_path: Path) -> None:
    path, _, _ = _checkpoint(tmp_path)

    with pytest.raises(ValueError, match="Unknown INT8 ConvRot gradient precision"):
        load_qwen_image_2_1_transformer(path, device=CPU, grad_precision="fp8")


def test_the_loaded_model_runs_a_forward_pass(tmp_path: Path) -> None:
    path, _, _ = _checkpoint(tmp_path)
    model = load_qwen_image_2_1_transformer(path, device=CPU, dtype=torch.float32)
    latents, text, timestep = tiny_inputs()

    out = model(latents, text, timestep, latent_h=2, latent_w=3)

    assert out.shape == (2, 6, 4)
    assert torch.isfinite(out).all()


def test_a_checkpoint_missing_a_parameter_fails_the_strict_load(tmp_path: Path) -> None:
    broken = {k: v for k, v in tiny_transformer().state_dict().items() if k != "proj_out.weight"}
    path = tmp_path / "broken.safetensors"
    int8_checkpoint(path, broken, {"img_in": 4})

    with pytest.raises(RuntimeError, match=re.escape("proj_out.weight")):
        load_qwen_image_2_1_transformer(path, device=CPU)


def test_a_quantized_tensor_for_a_module_the_model_lacks_is_rejected(tmp_path: Path) -> None:
    source = tiny_transformer()
    state = {**source.state_dict(), "bogus.weight": torch.zeros(4, 4)}
    path = tmp_path / "bogus.safetensors"
    int8_checkpoint(path, state, {"bogus": 4})

    with pytest.raises(ValueError, match="targets 1 linears"):
        load_qwen_image_2_1_transformer(path, device=CPU)


def test_a_checkpoint_the_state_dict_reader_reports_as_unquantized_loads_as_a_plain_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tiny_transformer(seed=7)
    path = tmp_path / "plain.safetensors"
    save_file(source.state_dict(), str(path))
    monkeypatch.setattr(model_module, "load_comfy_int8_convrot_state_dict", lambda p, **_: (load_file(p), 0))

    model = load_qwen_image_2_1_transformer(path, device=CPU, dtype=torch.float32)

    torch.testing.assert_close(model.img_in.weight, source.img_in.weight)
    assert not hasattr(model.img_in, "scale_weight")
    assert not any(p.requires_grad for p in model.parameters())
