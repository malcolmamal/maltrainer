from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import Qwen2Tokenizer

from maltrainer.models.minimax_h3.int8_convrot import rotate_activation
from maltrainer.models.moss_image.text_encoder import (
    TEMPLATE,
    TOKENIZER_ASSETS,
    MOSSImageTextEncoder,
    load_moss_image_text_encoder,
)
from maltrainer.models.qwen_image_2_1.text_encoder import Int8Embedding

from .qwen35_tiny import GROUP, HIDDEN, quantized_tensors, tiny_text_model, write_text_checkpoint

CPU = torch.device("cpu")
PROMPT_IDS = torch.tensor([[1, 5, 9, 2, 7, 3]])


def _load(path: Path, **kwargs) -> MOSSImageTextEncoder:
    return load_moss_image_text_encoder(path, CPU, **kwargs)


@pytest.mark.parametrize("prefix", ["", "model.", "model.language_model.", "text_encoders.qwen35_9b.transformer."])
def test_a_plain_checkpoint_loads_under_any_known_prefix_and_reproduces_the_source_model(
    tmp_path: Path, prefix: str
) -> None:
    source = tiny_text_model()
    path = write_text_checkpoint(source.state_dict(), tmp_path / "te.safetensors", prefix)

    encoder = _load(path, dtype=torch.float32)

    assert isinstance(encoder, MOSSImageTextEncoder)
    assert encoder.model.config == source.config
    assert encoder.model.compute_dtype == torch.float32
    torch.testing.assert_close(encoder.model(PROMPT_IDS), source(PROMPT_IDS))
    assert not any(p.requires_grad for p in encoder.model.parameters())
    assert not encoder.model.training


def test_the_loader_uses_the_vendored_qwen_tokenizer_by_default(tmp_path: Path) -> None:
    path = write_text_checkpoint(tiny_text_model().state_dict(), tmp_path / "te.safetensors")

    encoder = _load(path)

    assert isinstance(encoder.tokenizer, Qwen2Tokenizer)
    text = TEMPLATE.format("a red cat")
    assert encoder.tokenizer.decode(encoder.tokenizer(text, add_special_tokens=False).input_ids) == text
    assert (TOKENIZER_ASSETS / "vocab.json").exists()


def test_weights_are_stored_in_the_requested_dtype_while_activations_stay_fp32(tmp_path: Path) -> None:
    source = tiny_text_model()
    path = write_text_checkpoint(source.state_dict(), tmp_path / "te.safetensors")

    encoder = _load(path, dtype=torch.bfloat16)
    model = encoder.model

    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    out = model(PROMPT_IDS)
    assert out.dtype == torch.float32
    reference = source(PROMPT_IDS)
    assert torch.nn.functional.cosine_similarity(out.flatten(), reference.flatten(), dim=0) > 0.98


def test_the_compute_dtype_is_what_the_model_runs_its_activations_in(tmp_path: Path) -> None:
    path = write_text_checkpoint(tiny_text_model().state_dict(), tmp_path / "te.safetensors")

    encoder = _load(path, dtype=torch.bfloat16, compute_dtype=torch.bfloat16)

    assert encoder.model.compute_dtype == torch.bfloat16
    assert encoder.model(PROMPT_IDS).dtype == torch.bfloat16


def test_vision_head_and_mtp_tensors_are_ignored_on_load(tmp_path: Path) -> None:
    source = tiny_text_model()
    tensors = dict(source.state_dict())
    tensors.update(
        {"visual.blocks.0.w": torch.zeros(2), "lm_head.weight": torch.zeros(3), "mtp.fc.weight": torch.ones(1)}
    )
    path = write_text_checkpoint(tensors, tmp_path / "te.safetensors", "model.")

    encoder = _load(path, dtype=torch.float32)

    torch.testing.assert_close(encoder.model(PROMPT_IDS), source(PROMPT_IDS))


def test_a_missing_checkpoint_is_reported_by_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="text encoder not found"):
        _load(tmp_path / "nope.safetensors")


def test_a_checkpoint_that_is_not_a_qwen35_text_encoder_is_refused_before_building(tmp_path: Path) -> None:
    path = write_text_checkpoint({"layers.0.foo": torch.zeros(1)}, tmp_path / "other.safetensors")

    with pytest.raises(ValueError, match="Not a Qwen3.5 text encoder"):
        _load(path)


def test_a_checkpoint_with_an_unknown_extra_tensor_is_refused_naming_it(tmp_path: Path) -> None:
    tensors = dict(tiny_text_model().state_dict())
    tensors["layers.0.surprise.weight"] = torch.zeros(1)
    path = write_text_checkpoint(tensors, tmp_path / "te.safetensors")

    with pytest.raises(ValueError, match="does not look like a Qwen3.5 text encoder.*surprise"):
        _load(path)


def test_a_checkpoint_missing_a_weight_is_refused_naming_it(tmp_path: Path) -> None:
    tensors = dict(tiny_text_model().state_dict())
    del tensors["layers.1.mlp.up_proj.weight"]
    path = write_text_checkpoint(tensors, tmp_path / "te.safetensors")

    with pytest.raises(ValueError, match=r"missing \['layers.1.mlp.up_proj.weight'\]"):
        _load(path)


def test_an_int8_convrot_checkpoint_keeps_int8_linears_and_unrotates_its_embedding(tmp_path: Path) -> None:
    source = tiny_text_model()
    path = write_text_checkpoint(quantized_tensors(source), tmp_path / "te_int8.safetensors", "model.")

    encoder = _load(path, dtype=torch.float32)
    model = encoder.model

    assert isinstance(model.embed_tokens, Int8Embedding)
    assert int(model.embed_tokens.int8_convrot_groupsize) == GROUP
    assert model.layers[0].mlp.up_proj.weight.dtype == torch.int8
    assert model.layers[0].mlp.up_proj.int8_convrot_grad_precision == "bf16"
    assert model.layers[0].linear_attn.in_proj_qkv.weight.dtype == torch.float32
    model(PROMPT_IDS)
    assert model.embed_tokens.out_dtype == torch.float32  # the model sets the lookup's dtype each forward
    embedded = model.embed_tokens(PROMPT_IDS)
    torch.testing.assert_close(embedded, source.embed_tokens(PROMPT_IDS), atol=0.01, rtol=0.05)
    # the stored table really is rotated: a lookup that skipped the un-rotation would differ
    stored = model.embed_tokens.weight.float() * model.embed_tokens.scale_weight
    assert not torch.allclose(stored[PROMPT_IDS[0]], source.embed_tokens(PROMPT_IDS)[0], atol=0.05)
    torch.testing.assert_close(rotate_activation(stored, GROUP)[PROMPT_IDS[0]], embedded[0])


def test_an_int8_convrot_encoder_tracks_the_float_model_it_was_quantized_from(tmp_path: Path) -> None:
    source = tiny_text_model()
    path = write_text_checkpoint(quantized_tensors(source), tmp_path / "te_int8.safetensors")

    out = _load(path, dtype=torch.float32).model(PROMPT_IDS)

    reference = source(PROMPT_IDS)
    assert out.shape == reference.shape
    assert torch.nn.functional.cosine_similarity(out.flatten(), reference.flatten(), dim=0) > 0.95
    assert not torch.equal(out, reference)  # the int8 path really ran


def test_an_int8_checkpoint_with_an_unquantized_embedding_keeps_a_plain_embedding(tmp_path: Path) -> None:
    source = tiny_text_model()
    tensors = quantized_tensors(source)
    for suffix in ("weight_scale", "comfy_quant"):
        del tensors[f"embed_tokens.{suffix}"]
    tensors["embed_tokens.weight"] = source.embed_tokens.weight.detach()
    path = write_text_checkpoint(tensors, tmp_path / "te_int8.safetensors")

    encoder = _load(path, dtype=torch.float32)

    assert isinstance(encoder.model.embed_tokens, torch.nn.Embedding)
    assert encoder.model.embed_tokens.weight.shape[1] == HIDDEN
    assert encoder.model.layers[0].mlp.gate_proj.weight.dtype == torch.int8
