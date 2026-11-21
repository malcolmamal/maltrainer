from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.fibo2.loading import load_fibo2_transformer, strip_container_prefix

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


def test_a_dev_int8_file_is_decoded_to_dense_weights_by_default(tmp_path: Path) -> None:
    from maltrainer.models.minimax_h3.int8_rotation import dequantize_weight

    from .int8_files import GROUP, int8_state, write_int8

    path, stored = write_int8(tmp_path / "fibo2_dev_int8_convrot.safetensors")
    tensors, _ = int8_state()

    model, info = load_fibo2_transformer(path, CPU, dtype=torch.float32)

    assert len(stored) == (6 + 1) * 4  # attention (qkv, out) and FFN (w13, w2) of the 6 layers and the refiner
    assert not any(hasattr(module, "scale_weight") for module in model.modules())
    for base in stored:
        expected = dequantize_weight(tensors[f"{base}.weight"], tensors[f"{base}.weight_scale"], GROUP, torch.float32)
        assert torch.equal(model.get_parameter(f"{base}.weight"), expected)
    assert torch.equal(model.get_parameter("x_embedder.weight"), tiny_transformer().get_parameter("x_embedder.weight"))
    assert info["format"] == "int8_convrot" and info["int8_load"] == "decode" and info["int8_layers"] == len(stored)


def test_a_decoded_dev_int8_file_computes_what_the_dense_model_computes_to_int8_precision(tmp_path: Path) -> None:
    from .int8_files import write_int8

    path, _ = write_int8(tmp_path / "fibo2_dev_int8_convrot.safetensors")

    model, _ = load_fibo2_transformer(path, CPU, dtype=torch.float32)

    reference, got = tiny_transformer()(*tiny_inputs()), model(*tiny_inputs())
    assert (got - reference).norm() / reference.norm() < 0.05


def test_the_runtime_mode_keeps_the_linears_int8_with_their_scales(tmp_path: Path) -> None:
    from .int8_files import GROUP, write_int8

    path, stored = write_int8(tmp_path / "fibo2_dev_int8_convrot.safetensors")

    model, info = load_fibo2_transformer(path, CPU, int8_load="runtime")

    for base in stored:
        module = model.get_submodule(base)
        assert module.weight.dtype == torch.int8 and module.scale_weight.dtype == torch.float32
        assert int(module.int8_convrot_groupsize) == GROUP
    assert model.get_parameter("x_embedder.weight").dtype == torch.bfloat16
    assert info["int8_load"] == "runtime" and info["int8_layers"] == len(stored)


def test_an_edit_turbo_int8_file_is_refused_because_it_is_distilled(tmp_path: Path) -> None:
    from .int8_files import write_int8

    path, _ = write_int8(tmp_path / "fibo2_edit_turbo_int8_convrot.safetensors")

    with pytest.raises(ValueError, match=r"Edit Turbo.*distilled.*fibo2_dev_bf16"):
        load_fibo2_transformer(path, CPU)


def test_an_unknown_int8_mode_is_refused(tmp_path: Path) -> None:
    from .int8_files import write_int8

    path, _ = write_int8(tmp_path / "fibo2_dev_int8_convrot.safetensors")

    with pytest.raises(ValueError, match=r"int8_load must be one of \('decode', 'runtime'\), got 'half'"):
        load_fibo2_transformer(path, CPU, int8_load="half")


def test_a_marker_that_is_not_int8_convrot_is_refused(tmp_path: Path) -> None:
    from .int8_files import int8_state

    tensors, stored = int8_state()
    tensors[f"{stored[0]}.comfy_quant"] = torch.tensor(list(b'{"format": "fp8"}'), dtype=torch.uint8)
    path = write(tmp_path / "fibo2_dev_int8_convrot.safetensors", tensors)

    with pytest.raises(ValueError, match="not an INT8 ConvRot marker"):
        load_fibo2_transformer(path, CPU)


def test_a_bf16_edit_turbo_file_still_loads_with_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write(tmp_path / "fibo2_edit_turbo_bf16.safetensors", tiny_transformer().state_dict())

    with caplog.at_level(logging.WARNING):
        load_fibo2_transformer(path, CPU)

    assert "distilled" in caplog.text
