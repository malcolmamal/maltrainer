from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from maltrainer.models.flux3.loading import load_flux3_transformer
from maltrainer.models.flux3.model import Flux3Config
from maltrainer.models.minimax_h3.int8_convrot import rotate_activation
from tests.maltrainer.models.flux3.checkpoints import GROUP_SIZE, int8_state, marker, plain_state, write
from tests.maltrainer.models.flux3.tiny_model import tiny_inputs, tiny_transformer

CPU = torch.device("cpu")


def test_a_plain_checkpoint_loads_into_a_model_that_computes_what_the_source_computes(tmp_path: Path) -> None:
    path = write(tmp_path / "klein.safetensors", plain_state())

    model, _ = load_flux3_transformer(path, CPU, dtype=torch.float32)

    inputs = tiny_inputs()
    torch.testing.assert_close(model(*inputs), tiny_transformer()(*inputs))


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "model.diffusion_model."])
def test_the_container_prefix_of_the_file_does_not_matter(tmp_path: Path, prefix: str) -> None:
    path = write(tmp_path / "klein.safetensors", plain_state(), prefix=prefix)

    model, _ = load_flux3_transformer(path, CPU, dtype=torch.float32)

    source = tiny_transformer().state_dict()
    assert all(torch.equal(model.state_dict()[key], value) for key, value in source.items())


def test_the_model_is_frozen_in_eval_mode_and_computes_in_the_requested_dtype(tmp_path: Path) -> None:
    path = write(tmp_path / "klein.safetensors", plain_state())

    model, _ = load_flux3_transformer(str(path), CPU)

    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert model.compute_dtype == torch.bfloat16
    assert model(*tiny_inputs()).dtype == torch.bfloat16


def test_the_summary_describes_a_plain_file_by_its_stored_dtype(tmp_path: Path) -> None:
    path = write(tmp_path / "klein.safetensors", plain_state(dtype=torch.bfloat16))

    _, info = load_flux3_transformer(path, CPU)

    assert info == {
        "file": "klein.safetensors",
        "format": "bfloat16",
        "int8_layers": 0,
        "hidden_size": 128,
        "depth": 1,
        "depth_single_blocks": 2,
        "fused_qkv": False,
        "inference_attention": {},
    }


def test_a_fused_qkv_checkpoint_is_detected_from_its_keys(tmp_path: Path) -> None:
    config = Flux3Config(hidden_size=128, num_heads=1, depth=1, depth_single_blocks=1, fused_qkv=True)
    path = write(tmp_path / "fused.safetensors", plain_state(config))

    model, info = load_flux3_transformer(path, CPU)

    assert info["fused_qkv"] is True
    assert model.single_blocks[0].fused_qkv


def test_attention_hints_in_the_file_are_reported_but_not_loaded(tmp_path: Path) -> None:
    state = plain_state()
    state["single_blocks.0.comfy_attention.config"] = marker(attention="sol")
    path = write(tmp_path / "klein.safetensors", state)

    _, info = load_flux3_transformer(path, CPU)

    assert info["inference_attention"] == {"sol": "1 blocks"}


def test_a_missing_file_is_reported_with_its_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="FLUX.3 transformer not found"):
        load_flux3_transformer(tmp_path / "absent.safetensors", CPU)


def test_the_step_distilled_checkpoint_is_refused(tmp_path: Path) -> None:
    config = Flux3Config(hidden_size=128, num_heads=1, depth=1, depth_single_blocks=1, flow_map=True)
    path = write(tmp_path / "klein_sd.safetensors", plain_state(config))

    with pytest.raises(ValueError, match="klein_sd.safetensors is the step-distilled FLUX.3 checkpoint"):
        load_flux3_transformer(path, CPU)


def test_a_checkpoint_with_a_stray_tensor_does_not_load(tmp_path: Path) -> None:
    state = plain_state()
    state["single_blocks.0.bogus.weight"] = torch.zeros(1)
    path = write(tmp_path / "klein.safetensors", state)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_flux3_transformer(path, CPU)


def test_a_file_that_is_not_a_flux3_dit_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path / "other.safetensors", {"unet.weight": torch.zeros(1)})

    with pytest.raises(ValueError, match="Not a FLUX.3 DiT checkpoint"):
        load_flux3_transformer(path, CPU)


def test_an_int8_checkpoint_keeps_its_block_weights_in_int8_with_row_scales(tmp_path: Path) -> None:
    path = write(tmp_path / "int8.safetensors", int8_state(), prefix="model.diffusion_model.")

    model, info = load_flux3_transformer(path, CPU, dtype=torch.float32)

    block = model.single_blocks[1]
    assert block.q_proj.weight.dtype == torch.int8
    assert block.q_proj.scale_weight.shape == (128, 1)
    assert block.mlp_in.scale_weight.shape == (2 * 384, 1)
    assert int(block.q_proj.int8_convrot_groupsize) == GROUP_SIZE
    assert block.norm.query_norm.weight.dtype == torch.float32
    assert (info["format"], info["int8_layers"]) == ("int8_convrot", 5 * 6)


def test_an_int8_linear_rotates_its_input_and_applies_the_row_scales(tmp_path: Path) -> None:
    path = write(tmp_path / "int8.safetensors", int8_state())
    model, _ = load_flux3_transformer(path, CPU, dtype=torch.float32)
    linear = model.txt_mode_blocks[0].attn_out
    x = torch.randn(3, 128, generator=torch.Generator().manual_seed(1))

    out = linear(x)

    expected = F.linear(rotate_activation(x, GROUP_SIZE), linear.weight.float() * linear.scale_weight)
    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-4)


def test_an_int8_model_runs_end_to_end_and_takes_the_requested_gradient_precision(tmp_path: Path) -> None:
    path = write(tmp_path / "int8.safetensors", int8_state())

    model, _ = load_flux3_transformer(path, CPU, dtype=torch.float32, grad_precision="fp32")

    assert model.single_blocks[0].q_proj.int8_convrot_grad_precision == "fp32"
    out = model(*tiny_inputs())
    assert out.shape == (2, 128, 2, 3)
    assert torch.isfinite(out).all()


def test_a_dense_file_is_not_given_the_int8_runtime(tmp_path: Path) -> None:
    path = write(tmp_path / "klein.safetensors", plain_state())

    model, _ = load_flux3_transformer(path, CPU)

    assert not hasattr(model.single_blocks[0].q_proj, "scale_weight")
