from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import T5TokenizerFast

from maltrainer.models.supra2_img.model import MAX_CTX_LEN
from maltrainer.models.supra2_img.text_encoder import FlanT5Encoder, load_flan_t5

from ._tiny_t5 import tiny_config, tiny_t5, write_tokenizer, write_weights

CPU = torch.device("cpu")


def _encoder(tmp_path: Path, ctx_len: int = 6) -> FlanT5Encoder:
    tokenizer = T5TokenizerFast(tokenizer_file=str(write_tokenizer(tmp_path / "tok.json")), extra_ids=100)
    return FlanT5Encoder(tiny_t5(), tokenizer, ctx_len=ctx_len)


def test_the_default_context_length_is_the_models_maximum() -> None:
    assert FlanT5Encoder(tiny_t5(), tokenizer=None).ctx_len == MAX_CTX_LEN == 128


def test_prompts_are_padded_to_the_context_length_with_a_matching_float_mask(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)

    hidden, mask = encoder.encode(["a red cat", "cat"], CPU)

    assert hidden.shape == (2, 6, 8) and hidden.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert mask.sum(dim=1).tolist() == [4.0, 2.0]
    assert mask[0].tolist() == [1, 1, 1, 1, 0, 0]  # three words plus the eos T5 appends


def test_the_hidden_states_are_the_t5_encoder_output_for_those_tokens(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path)
    tokens = encoder.tokenizer(["a red cat"], padding="max_length", max_length=6, return_tensors="pt")

    hidden, _ = encoder.encode(["a red cat"], CPU)

    with torch.no_grad():
        expected = encoder.model(**tokens).last_hidden_state
    assert torch.allclose(hidden, expected)
    assert not hidden.requires_grad


def test_prompts_longer_than_the_context_are_truncated(tmp_path: Path) -> None:
    encoder = _encoder(tmp_path, ctx_len=3)

    hidden, mask = encoder.encode(["a red cat sat on the mat"], CPU)

    assert hidden.shape[1] == 3 and mask.tolist() == [[1.0, 1.0, 1.0]]


def test_different_prompts_give_different_conditioning(tmp_path: Path) -> None:
    hidden, _ = _encoder(tmp_path).encode(["a red cat", "a mat"], CPU)

    assert not torch.allclose(hidden[0, 0], hidden[1, 1])


def _files(tmp_path: Path, **kwargs: object) -> tuple[Path, Path, Path]:
    model = tiny_t5()
    config = tmp_path / "config.json"
    tiny_config().to_json_file(str(config))
    return write_weights(tmp_path / "w.safetensors", model, **kwargs), config, write_tokenizer(tmp_path / "tok.json")


def test_the_loader_returns_a_frozen_eval_encoder_that_reproduces_the_saved_weights(tmp_path: Path) -> None:
    weights, config, tokenizer = _files(tmp_path)

    encoder = load_flan_t5(weights, config, tokenizer, CPU, ctx_len=5)

    assert encoder.ctx_len == 5 and not encoder.model.training
    assert not any(p.requires_grad for p in encoder.model.parameters())
    reference = tiny_t5()
    for (name, loaded), (_, original) in zip(
        encoder.model.state_dict().items(), reference.state_dict().items(), strict=False
    ):
        assert torch.equal(loaded, original), name


def test_embedding_tables_may_be_absent_from_the_file_because_they_are_tied(tmp_path: Path) -> None:
    weights, config, tokenizer = _files(tmp_path, drop=("encoder.embed_tokens",))

    encoder = load_flan_t5(weights, config, tokenizer, CPU)

    assert isinstance(encoder, FlanT5Encoder)


def test_decoder_and_lm_head_tensors_are_tolerated_as_unexpected_keys(tmp_path: Path) -> None:
    extra = {"decoder.block.0.weight": torch.zeros(1), "lm_head.weight": torch.zeros(1)}
    weights, config, tokenizer = _files(tmp_path, extra=extra)

    assert isinstance(load_flan_t5(weights, config, tokenizer, CPU), FlanT5Encoder)


def test_an_unrelated_tensor_in_the_file_is_reported_as_unexpected(tmp_path: Path) -> None:
    weights, config, tokenizer = _files(tmp_path, extra={"mystery.weight": torch.zeros(1)})

    with pytest.raises(RuntimeError, match=r"unexpected=\['mystery.weight'\]"):
        load_flan_t5(weights, config, tokenizer, CPU)


def test_a_missing_encoder_weight_is_reported(tmp_path: Path) -> None:
    weights, config, tokenizer = _files(tmp_path, drop=("encoder.final_layer_norm",))

    with pytest.raises(RuntimeError, match=r"missing=\['encoder.final_layer_norm.weight'\]"):
        load_flan_t5(weights, config, tokenizer, CPU)
