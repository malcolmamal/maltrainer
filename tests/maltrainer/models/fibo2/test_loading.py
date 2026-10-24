from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.fibo2.loading import is_quantized, load_fibo2_transformer, strip_container_prefix

from .tiny_model import TINY, tiny_inputs, tiny_transformer

CPU = torch.device("cpu")


def write(path: Path, state: dict[str, torch.Tensor], prefix: str = "") -> Path:
    save_file({prefix + key: value.contiguous() for key, value in state.items()}, str(path))
    return path


def test_a_checkpoint_loads_into_a_model_that_computes_what_the_source_computes(tmp_path: Path) -> None:
    path = write(tmp_path / "fibo.safetensors", tiny_transformer().state_dict())

    model, _ = load_fibo2_transformer(path, CPU, dtype=torch.float32)

    inputs = tiny_inputs()
    torch.testing.assert_close(model(*inputs), tiny_transformer()(*inputs))
    assert model.config == TINY


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "model.diffusion_model."])
def test_the_container_prefix_of_the_file_does_not_matter(tmp_path: Path, prefix: str) -> None:
    path = write(tmp_path / "fibo.safetensors", tiny_transformer().state_dict(), prefix=prefix)

    model, _ = load_fibo2_transformer(path, CPU, dtype=torch.float32)

    source = tiny_transformer().state_dict()
    assert all(torch.equal(model.state_dict()[key], value) for key, value in source.items())


def test_a_prefix_only_some_keys_carry_is_left_alone() -> None:
    state = {"diffusion_model.a": torch.zeros(1), "b": torch.zeros(1)}

    assert strip_container_prefix(state) is state


def test_the_model_is_frozen_in_eval_mode_and_computes_in_the_requested_dtype(tmp_path: Path) -> None:
    path = write(tmp_path / "fibo.safetensors", tiny_transformer().state_dict())

    model, _ = load_fibo2_transformer(str(path), CPU)

    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert model.compute_dtype == torch.bfloat16
    assert model(*tiny_inputs()).dtype == torch.bfloat16


def test_the_summary_names_the_file_and_the_stored_dtype(tmp_path: Path) -> None:
    state = {key: value.bfloat16() for key, value in tiny_transformer().state_dict().items()}
    path = write(tmp_path / "fibo.safetensors", state)

    _, info = load_fibo2_transformer(path, CPU)

    assert info == {
        "file": "fibo.safetensors",
        "format": "bfloat16",
        "dim": 256,
        "layers": 6,
        "injection_layers": [1, 2, 3, 4, 5],
    }


def test_a_missing_file_is_reported_by_name(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"Fibo 2\.0 transformer not found at .*nope\.safetensors"):
        load_fibo2_transformer(tmp_path / "nope.safetensors", CPU)


def test_an_int8_convrot_file_is_refused_with_the_reason(tmp_path: Path) -> None:
    state = tiny_transformer().state_dict()
    state["layers.0.attention.qkv.comfy_quant"] = torch.zeros(4, dtype=torch.uint8)
    path = write(tmp_path / "fibo2_edit_turbo_int8_convrot.safetensors", state)

    assert is_quantized(path)
    with pytest.raises(ValueError, match=r"Edit Turbo, which is\s+distilled.*fibo2_dev_bf16"):
        load_fibo2_transformer(path, CPU)


def test_a_file_missing_a_tensor_does_not_load(tmp_path: Path) -> None:
    state = tiny_transformer().state_dict()
    del state["final_layer.linear.bias"]
    path = write(tmp_path / "fibo.safetensors", state)

    with pytest.raises(RuntimeError, match=r"final_layer\.linear\.bias"):
        load_fibo2_transformer(path, CPU)


def test_loading_is_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write(tmp_path / "fibo.safetensors", tiny_transformer().state_dict())

    with caplog.at_level(logging.INFO, logger="maltrainer.models.fibo2.loading"):
        load_fibo2_transformer(path, CPU)

    assert "Fibo 2.0 DiT ready" in caplog.text
