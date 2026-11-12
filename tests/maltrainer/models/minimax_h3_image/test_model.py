"""The H3 Image transformer: no audio stream, and one output head per noise level."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from maltrainer.models.minimax_h3_image.grid import PDD_SIGMAS, file_pdd_sigmas
from maltrainer.models.minimax_h3_image.model import MiniMaxH3ImageTransformer
from tests.maltrainer.models.minimax_h3.h3_tiny import Packed, tiny_config

LEVELS = 6
PATCH = 8  # in_channels 2 x a 1x2x2 patch


def _model(seed: int = 0) -> MiniMaxH3ImageTransformer:
    torch.manual_seed(seed)
    return MiniMaxH3ImageTransformer(tiny_config(), LEVELS).eval()


def _inputs(packed: Packed) -> dict[str, torch.Tensor]:
    inputs = packed.inputs()
    for audio_only in ("audio_hidden_states", "audio_indices"):
        del inputs[audio_only]
    inputs["timestep"] = inputs["timestep"][:1]
    return inputs


def test_the_checkpoint_layout_has_no_audio_tensors_and_one_stacked_head_bank() -> None:
    state = _model().state_dict()

    assert not [key for key in state if "audio" in key]
    assert state["final_layer.video_out.weight"].shape == (LEVELS * PATCH, 16)
    assert state["final_layer.video_out.bias"].shape == (LEVELS * PATCH,)


def test_a_forward_returns_one_row_per_image_token() -> None:
    packed = Packed(text=3, audio=0, video=4)

    out = _model()(level=2, **_inputs(packed))

    assert out.shape == (1, 4, PATCH)


def test_the_head_for_a_level_is_the_base_head_plus_that_levels_offset() -> None:
    final = _model().final_layer
    rows = final.video_out.weight.view(LEVELS, PATCH, -1)
    bias = final.video_out.bias.view(LEVELS, PATCH)

    assert torch.equal(final.head(0)[0], rows[0]) and torch.equal(final.head(0)[1], bias[0])
    for level in range(1, LEVELS):
        weight, offset = final.head(level)
        assert torch.equal(weight, rows[0] + rows[level]) and torch.equal(offset, bias[0] + bias[level])


def test_every_level_predicts_through_its_own_head() -> None:
    model = _model()
    inputs = _inputs(Packed(text=3, audio=0, video=4))

    with torch.no_grad():
        outputs = [model(level=level, **inputs) for level in range(LEVELS)]

    assert all(not torch.allclose(outputs[0], other) for other in outputs[1:])


@pytest.mark.parametrize("level", [-1, LEVELS])
def test_a_level_outside_the_head_bank_is_refused(level: int) -> None:
    with pytest.raises(ValueError, match="output heads"):
        _model().final_layer.head(level)


def test_the_grid_ends_at_zero_and_falls() -> None:
    assert PDD_SIGMAS[0] == 1.0 and PDD_SIGMAS[-1] == 0.0
    assert list(PDD_SIGMAS) == sorted(PDD_SIGMAS, reverse=True)


def _write_header(path: Path, metadata: dict[str, str] | bytes) -> Path:
    header = metadata if isinstance(metadata, bytes) else json.dumps({"__metadata__": metadata}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


def test_the_grid_is_read_from_the_files_metadata(tmp_path: Path) -> None:
    config = json.dumps({"transformer": {"pdd_sigmas": [1.0, 0.5, 0.0]}})

    assert file_pdd_sigmas(_write_header(tmp_path / "a.safetensors", {"config": config})) == (1.0, 0.5, 0.0)


@pytest.mark.parametrize(
    "metadata",
    [{}, {"config": "{}"}, {"config": "not json"}, {"config": json.dumps({"transformer": {"pdd_sigmas": [1.0]}})}],
)
def test_a_file_that_does_not_state_its_grid_gets_the_released_one(tmp_path: Path, metadata: dict[str, str]) -> None:
    assert file_pdd_sigmas(_write_header(tmp_path / "b.safetensors", metadata)) == PDD_SIGMAS


def test_a_missing_or_unreadable_file_gets_the_released_grid(tmp_path: Path) -> None:
    assert file_pdd_sigmas(tmp_path / "absent.safetensors") == PDD_SIGMAS
    assert file_pdd_sigmas(_write_header(tmp_path / "bad.safetensors", b"{broken")) == PDD_SIGMAS


def test_the_final_modulation_is_applied_in_float32_as_comfyui_does() -> None:
    final = _model().final_layer
    final.norm.to(torch.bfloat16)  # the checkpoint stores its norms in bf16
    with torch.no_grad():
        final.adaln_proj.linear.bias.copy_(torch.linspace(0.1234567, 0.7654321, final.adaln_proj.linear.bias.numel()))
    hidden = torch.randn(1, 5, 16).to(torch.bfloat16)
    timestep_embedding = torch.randn(1, 10)
    indices = torch.zeros(5, dtype=torch.long)
    video = torch.arange(5)

    with torch.no_grad():
        out = final(hidden, timestep_embedding, (indices, video), 2)
        shift, scale = final.adaln_proj(timestep_embedding).chunk(2, dim=-1)
        modulated = final.norm(hidden).float() * (1.0 + scale.float()) + shift.float()
        weight, bias = final.head(2)
        expected = torch.nn.functional.linear(modulated, weight, bias)

    torch.testing.assert_close(out, expected, rtol=0, atol=1e-6)
