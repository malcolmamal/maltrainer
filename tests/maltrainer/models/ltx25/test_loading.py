from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.ltx25.loading import drop_audio, load_ltx25_transformer
from tests.maltrainer.models.flux3.checkpoints import GROUP_SIZE, marker

from .tiny_model import tiny_inputs, tiny_transformer

CPU = torch.device("cpu")
CONFIG = json.dumps(
    {
        "transformer": {
            "positional_embedding_max_pos": [20, 64, 64],
            "num_attention_heads": 2,
            "audio_num_attention_heads": 2,
            "connector_attention_head_dim": 48,
            "audio_connector_attention_head_dim": 24,
        }
    }
)


def write(path: Path, state: dict[str, torch.Tensor], prefix: str = "") -> Path:
    save_file(
        {prefix + key: value.contiguous() for key, value in state.items()}, str(path), metadata={"config": CONFIG}
    )
    return path


def int8_state() -> dict[str, torch.Tensor]:
    """The block attention and feed-forward Linears as int8 with a row scale and a ConvRot marker."""
    generator = torch.Generator().manual_seed(5)
    state = dict(tiny_transformer().state_dict())
    for key in [
        k for k in state if k.startswith("transformer_blocks.") and k.endswith(".weight") and state[k].dim() == 2
    ]:
        base = key[: -len(".weight")]
        rows, cols = state[key].shape
        state[key] = torch.randint(-100, 100, (rows, cols), generator=generator, dtype=torch.int8)
        state[f"{base}.weight_scale"] = torch.full((rows,), 0.001)
        state[f"{base}.comfy_quant"] = marker(format="int8_tensorwise", convrot=True, convrot_groupsize=GROUP_SIZE)
    return state


def test_a_checkpoint_loads_into_a_model_that_computes_what_the_source_computes(tmp_path: Path) -> None:
    path = write(tmp_path / "ltx-2.5-22b-dev-transformer-bf16.safetensors", tiny_transformer().state_dict())

    model, info = load_ltx25_transformer(path, CPU, dtype=torch.float32)

    inputs = tiny_inputs()
    torch.testing.assert_close(
        model(*inputs[:3], 24.0, *inputs[3:]), tiny_transformer()(*inputs[:3], 24.0, *inputs[3:])
    )
    assert info == {"file": path.name, "format": "float32", "int8_layers": 0, "layers": 2, "audio": True}


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "model.diffusion_model."])
def test_the_container_prefix_of_the_file_does_not_matter(tmp_path: Path, prefix: str) -> None:
    path = write(tmp_path / "dev.safetensors", tiny_transformer().state_dict(), prefix)

    model, _ = load_ltx25_transformer(path, CPU, dtype=torch.float32)

    assert torch.equal(model.state_dict()["proj_out.weight"], tiny_transformer().state_dict()["proj_out.weight"])


def test_the_text_projection_that_rides_in_some_files_is_left_to_the_text_encoder(tmp_path: Path) -> None:
    state = tiny_transformer().state_dict()
    state["text_embedding_projection.video_aggregate_embed.weight"] = torch.zeros(2, 2)
    path = write(tmp_path / "dev.safetensors", state)

    model, _ = load_ltx25_transformer(path, CPU, dtype=torch.float32)

    assert not any(key.startswith("text_embedding_projection") for key in model.state_dict())


def test_the_model_is_frozen_in_eval_mode_and_computes_in_the_requested_dtype(tmp_path: Path) -> None:
    path = write(tmp_path / "dev.safetensors", tiny_transformer().state_dict())

    model, _ = load_ltx25_transformer(str(path), CPU)

    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16} and model.compute_dtype == torch.bfloat16


def test_without_audio_the_audio_tensors_of_the_file_are_not_loaded(tmp_path: Path) -> None:
    path = write(tmp_path / "dev.safetensors", tiny_transformer().state_dict())

    model, info = load_ltx25_transformer(path, CPU, dtype=torch.float32, with_audio=False)

    assert not model.with_audio and info["audio"] is False
    assert not any("audio" in key or "av_ca" in key or "a2v_ca" in key for key in model.state_dict())
    out, audio_out = model(*tiny_inputs()[:3])
    assert audio_out is None and out.shape[1] == 128


def test_drop_audio_removes_the_streams_modules_the_cross_modal_layers_and_their_scales() -> None:
    names = [
        "transformer_blocks.0.attn1.to_q.weight",
        "transformer_blocks.0.audio_attn1.to_q.weight",
        "transformer_blocks.0.audio_to_video_attn.to_q.weight",
        "transformer_blocks.0.scale_shift_table_a2v_ca_video",
        "av_ca_a2v_gate_adaln_single.linear.weight",
        "audio_embeddings_connector.learnable_registers",
        "video_embeddings_connector.learnable_registers",
    ]

    assert set(drop_audio({name: torch.zeros(1) for name in names})) == {names[0], names[-1]}


def test_a_distilled_checkpoint_is_refused_with_the_reason(tmp_path: Path) -> None:
    path = write(tmp_path / "ltx-2.5-22b-distilled-transformer-bf16.safetensors", tiny_transformer().state_dict())

    with pytest.raises(ValueError, match=r"distilled.*train the dev model"):
        load_ltx25_transformer(path, CPU)


def test_a_file_without_an_audio_stream_cannot_be_loaded_with_audio(tmp_path: Path) -> None:
    state = {
        k: v
        for k, v in tiny_transformer().state_dict().items()
        if not any(m in k for m in ("audio", "av_ca", "a2v_ca"))
    }
    path = write(tmp_path / "dev.safetensors", state)

    with pytest.raises(ValueError, match="no audio stream"):
        load_ltx25_transformer(path, CPU)
    assert load_ltx25_transformer(path, CPU, with_audio=False)[0].with_audio is False


def test_an_ltx_2_0_checkpoint_with_a_caption_projection_is_refused(tmp_path: Path) -> None:
    state = tiny_transformer().state_dict()
    state["caption_projection.linear_1.weight"] = torch.zeros(96, 96)
    path = write(tmp_path / "dev.safetensors", state)

    with pytest.raises(ValueError, match=r"caption projection.*LTX-2\.0"):
        load_ltx25_transformer(path, CPU)


def test_a_missing_file_is_reported_by_name(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"LTX-2\.5 transformer not found at .*gone\.safetensors"):
        load_ltx25_transformer(tmp_path / "gone.safetensors", CPU)


def test_an_int8_convrot_checkpoint_keeps_its_block_linears_int8_and_still_computes(tmp_path: Path) -> None:
    path = write(tmp_path / "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors", int8_state())

    model, info = load_ltx25_transformer(path, CPU, dtype=torch.float32)

    assert info["format"] == "int8_convrot" and info["int8_layers"] > 0
    assert model.get_parameter("transformer_blocks.0.attn1.to_q.weight").dtype == torch.int8
    assert model.get_parameter("patchify_proj.weight").dtype == torch.float32
    out, _ = model(*tiny_inputs()[:3])
    assert torch.isfinite(out).all()


def test_the_int8_layers_of_an_audio_free_load_are_only_the_video_ones(tmp_path: Path) -> None:
    path = write(tmp_path / "int8.safetensors", int8_state())

    with_audio = load_ltx25_transformer(path, CPU, dtype=torch.float32)[1]["int8_layers"]
    video_only = load_ltx25_transformer(path, CPU, dtype=torch.float32, with_audio=False)[1]["int8_layers"]

    assert 0 < video_only < with_audio
