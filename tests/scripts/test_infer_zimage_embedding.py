"""scripts/infer_zimage.py: loading a textual-inversion embedding into the Qwen tokenizer and encoder."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tests.scripts.zimage_script_fakes import (
    WORDS,
    Loader,
    loaded_infer,
    pipeline_class,
    tiny_text_encoder,
    tiny_tokenizer,
)

VECTOR = torch.arange(8, dtype=torch.float32).reshape(1, 8) / 4


@pytest.fixture
def infer(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    with loaded_infer(monkeypatch, pipeline_class()) as module:
        yield module


@pytest.fixture
def asked(monkeypatch: pytest.MonkeyPatch, infer: ModuleType) -> list[tuple[str, str]]:
    requests: list[tuple[str, str]] = []
    monkeypatch.setattr(infer, "Qwen2Tokenizer", Loader(tiny_tokenizer, requests))
    monkeypatch.setattr(infer, "Qwen3ForCausalLM", Loader(tiny_text_encoder, requests))
    return requests


def _file(path: Path, **tensors: torch.Tensor) -> str:
    save_file(tensors, str(path))
    return str(path)


def _pipeline() -> SimpleNamespace:
    return SimpleNamespace(device="cpu", tokenizer=None, text_encoder=None)


def test_a_new_token_is_added_and_its_row_takes_the_embedding_in_the_encoders_dtype(
    tmp_path: Path, infer: ModuleType, asked: list[tuple[str, str]]
) -> None:
    pipeline = _pipeline()

    infer.load_textual_inversion_embedding(pipeline, _file(tmp_path / "e.safetensors", embedding=VECTOR))

    table = pipeline.text_encoder.get_input_embeddings().weight
    assert len(pipeline.tokenizer) == len(WORDS) + 1 and table.shape == (len(WORDS) + 1, 8)
    assert pipeline.tokenizer.convert_tokens_to_ids("<*>") == len(WORDS)
    assert table.dtype == torch.bfloat16 and torch.equal(table[len(WORDS)], VECTOR[0].to(torch.bfloat16))
    assert asked == [("Tongyi-MAI/Z-Image", "tokenizer"), ("Tongyi-MAI/Z-Image", "text_encoder")]


def test_the_token_string_names_the_new_token(tmp_path: Path, infer: ModuleType, asked: list[tuple[str, str]]) -> None:
    pipeline = _pipeline()

    infer.load_textual_inversion_embedding(pipeline, _file(tmp_path / "e.safetensors", embedding=VECTOR), "<cat>")

    assert pipeline.tokenizer.convert_tokens_to_ids("<cat>") == len(WORDS) and asked


def test_a_token_the_tokenizer_already_knows_keeps_its_id_and_has_its_row_overwritten(
    tmp_path: Path, infer: ModuleType, asked: list[tuple[str, str]]
) -> None:
    pipeline = _pipeline()

    infer.load_textual_inversion_embedding(pipeline, _file(tmp_path / "e.safetensors", embedding=VECTOR), "photo")

    table = pipeline.text_encoder.get_input_embeddings().weight
    assert len(pipeline.tokenizer) == len(WORDS) and table.shape[0] == len(WORDS)
    assert torch.equal(table[WORDS.index("photo")], VECTOR[0].to(torch.bfloat16)) and asked


@pytest.mark.parametrize(
    ("tensors", "found"),
    [
        ({"emb_params": VECTOR, "embedding": VECTOR + 1}, VECTOR),
        ({"embedding": VECTOR + 1}, VECTOR + 1),
        ({"vectors": VECTOR + 2}, VECTOR + 2),
        ({"scalar": torch.ones(3), "big": torch.ones(9, 8), "vectors": VECTOR + 3}, VECTOR + 3),
    ],
)
def test_the_vectors_come_from_emb_params_then_embedding_then_the_first_small_2d_tensor(
    tmp_path: Path,
    infer: ModuleType,
    asked: list[tuple[str, str]],
    tensors: dict[str, torch.Tensor],
    found: torch.Tensor,
) -> None:
    pipeline = _pipeline()

    infer.load_textual_inversion_embedding(pipeline, _file(tmp_path / "e.safetensors", **tensors))

    table = pipeline.text_encoder.get_input_embeddings().weight
    assert torch.equal(table[len(WORDS)], found[0].to(torch.bfloat16)) and asked


def test_a_file_without_anything_that_looks_like_an_embedding_is_refused_before_any_model_loads(
    tmp_path: Path, infer: ModuleType, asked: list[tuple[str, str]]
) -> None:
    path = _file(tmp_path / "e.safetensors", scalar=torch.ones(3), big=torch.ones(9, 8))

    with pytest.raises(ValueError, match=r"Could not find embedding vectors in .*e\.safetensors"):
        infer.load_textual_inversion_embedding(_pipeline(), path)

    assert asked == []
