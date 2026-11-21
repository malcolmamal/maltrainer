from __future__ import annotations

from pathlib import Path

import pytest
import torch

from maltrainer.models.ltx25 import text_encoder as te
from maltrainer.models.ltx25.text_encoder import (
    GEMMA_DEFAULT,
    ScaledEmbedding,
    gemma_config,
    load_ltx25_text_encoder,
    text_keys,
)
from maltrainer.models.ltx25.text_projection import DualProjection, stack_hidden_states

from .text_files import AUDIO_DIM, GEMMA, VIDEO_DIM, write_text_encoder

CPU = torch.device("cpu")


def _encoder(tmp_path: Path, max_tokens: int = te.MAX_TOKENS) -> te.LTX25TextEncoder:
    return load_ltx25_text_encoder(
        write_text_encoder(tmp_path / "te.safetensors"), CPU, torch.float32, max_tokens=max_tokens
    )


def test_the_features_are_one_row_per_token_with_the_video_and_audio_halves(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)

    features = encoder.encode("a red fox", CPU, torch.float32)

    assert features.shape == (4, VIDEO_DIM + AUDIO_DIM)  # BOS + three words


def test_the_prompt_is_tokenised_behind_a_bos_token_with_no_template_and_stripped(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)

    assert encoder.token_ids("  a red fox \n") == [2, 3, 4, 5]
    assert encoder.token_ids("") == [2]
    assert encoder.token_ids("a zzz") == [2, 3, 0]


def test_the_tokens_are_cut_at_the_maximum_length(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path, max_tokens=3)

    assert encoder.token_ids("a red fox runs") == [2, 3, 4]
    assert encoder.encode("a red fox runs", CPU, torch.float32).shape[0] == 3


def test_an_empty_prompt_is_a_single_bos_token(tmp_path: Path) -> None:
    assert _encoder(tmp_path).encode("", CPU).shape == (1, VIDEO_DIM + AUDIO_DIM)


def test_the_features_are_the_projection_of_all_hidden_states(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    ids = torch.tensor([encoder.token_ids("the sea runs")])

    with torch.no_grad():
        hidden = encoder.stack.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        expected = encoder.stack.text_embedding_projection(stack_hidden_states(hidden))[0]

    assert len(hidden) == GEMMA["num_hidden_layers"] + 1
    assert torch.allclose(encoder.encode("the sea runs", CPU, torch.float32), expected)


def test_the_prompt_changes_the_features_and_the_same_prompt_gives_the_same_ones(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)

    first, again, other = (encoder.encode(p, CPU, torch.float32) for p in ("a red fox", "a red fox", "a red sea"))

    assert torch.equal(first, again) and not torch.allclose(first, other)


def test_the_output_defaults_to_bf16(tmp_path: Path) -> None:
    assert _encoder(tmp_path).encode("a", CPU).dtype == torch.bfloat16


def test_the_loader_freezes_the_encoder_and_casts_it(tmp_path: Path) -> None:
    encoder = load_ltx25_text_encoder(write_text_encoder(tmp_path / "te.safetensors"), CPU, torch.bfloat16)

    assert not encoder.stack.training and not any(p.requires_grad for p in encoder.stack.parameters())
    assert {p.dtype for p in encoder.stack.parameters()} == {torch.bfloat16}


def test_an_int8_convrot_file_gets_int8_linears_and_a_scaled_int8_embedding(tmp_path: Path) -> None:
    path = write_text_encoder(tmp_path / "int8.safetensors", int8=True)

    encoder = load_ltx25_text_encoder(path, CPU, torch.bfloat16)

    layer = encoder.stack.model.layers[0]
    assert layer.self_attn.q_proj.weight.dtype == torch.int8 and layer.self_attn.k_proj.weight.dtype == torch.bfloat16
    assert isinstance(encoder.stack.model.embed_tokens, ScaledEmbedding)
    assert encoder.stack.model.embed_tokens.scale == pytest.approx(GEMMA["hidden_size"] ** 0.5)
    assert torch.isfinite(encoder.encode("a red fox", CPU, torch.float32)).all()


def test_the_scaled_embedding_multiplies_the_lookup_by_its_scale() -> None:
    table = torch.nn.Embedding(5, 3)
    wrapped = ScaledEmbedding(table, 2.0)

    assert torch.equal(wrapped(torch.tensor([1, 4])), table(torch.tensor([1, 4])) * 2.0)


def test_the_text_decoder_keys_are_found_under_either_prefix_and_the_towers_are_dropped() -> None:
    raw = {
        "model.layers.0.mlp.up_proj.weight": 1,
        "model.language_model.layers.1.mlp.up_proj.weight": 2,
        "model.vision_tower.encoder.weight": 3,
        "model.audio_tower.weight": 4,
        "model.embed_vision.weight": 5,
        "text_embedding_projection.video_aggregate_embed.weight": 6,
        "tokenizer_json": 7,
        "lm_head.weight": 8,
    }

    assert text_keys(raw) == {
        "model.layers.0.mlp.up_proj.weight": 1,
        "model.layers.1.mlp.up_proj.weight": 2,
        "text_embedding_projection.video_aggregate_embed.weight": 6,
    }


def test_the_gemma_config_comes_from_the_files_metadata_or_the_12b_defaults(tmp_path: Path) -> None:
    with_config = write_text_encoder(tmp_path / "a.safetensors")
    without = write_text_encoder(tmp_path / "b.safetensors", with_config=False)

    assert gemma_config(with_config) == GEMMA  # the dtype entry is dropped
    assert gemma_config(without) == GEMMA_DEFAULT
    assert GEMMA_DEFAULT["hidden_size"] == 3840 and GEMMA_DEFAULT["num_hidden_layers"] == 48


def test_a_missing_file_is_reported_by_name(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"LTX-2\.5 text encoder not found at .*gone\.safetensors"):
        load_ltx25_text_encoder(tmp_path / "gone.safetensors", CPU)


def test_a_file_without_a_tokenizer_is_refused_with_the_name_to_get(tmp_path: Path) -> None:
    path = write_text_encoder(tmp_path / "te.safetensors", tokenizer=False)

    with pytest.raises(ValueError, match=r"no tokenizer.*gemma4-12b-with-proj-ltx-2\.5"):
        load_ltx25_text_encoder(path, CPU)


def test_a_file_with_a_tensor_the_decoder_lacks_is_refused(tmp_path: Path) -> None:
    path = write_text_encoder(tmp_path / "te.safetensors", extra={"model.layers.0.mystery.weight": torch.zeros(1)})

    with pytest.raises(ValueError, match=r"does not match the Gemma 4 12B text encoder.*mystery"):
        load_ltx25_text_encoder(path, CPU)


def test_the_projection_normalises_each_layer_and_scales_by_the_output_width() -> None:
    projection = DualProjection(4, 2, 1)
    with torch.no_grad():
        projection.video_aggregate_embed.weight.copy_(torch.eye(2, 4))
        projection.video_aggregate_embed.bias.zero_()
        projection.audio_aggregate_embed.weight.copy_(torch.ones(1, 4))
        projection.audio_aggregate_embed.bias.zero_()
    states = torch.ones(1, 1, 2, 2) * 3.0  # hidden 2, two layers: every value 3 -> unit RMS 1 after the norm

    out = projection(states)

    assert out.shape == (1, 1, 3)
    assert out[0, 0, :2].tolist() == pytest.approx([(2 / 2) ** 0.5] * 2, abs=1e-5)
    assert out[0, 0, 2].item() == pytest.approx(4 * (1 / 2) ** 0.5, abs=1e-5)
