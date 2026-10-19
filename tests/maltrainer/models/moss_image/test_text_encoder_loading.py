from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.moss_image.text_encoder import MOSSImageTextEncoder, load_moss_image_text_encoder
from maltrainer.models.qwen_image_2_1.text_encoder import Int8Embedding

from .tiny import CPU
from .tiny_text import HIDDEN, TINY, text_int8_tensors, tiny_text_model

IDS = torch.tensor([[3, 7, 1, 9, 4]])


@pytest.fixture
def tokenizer_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "tokenizer"
    folder.mkdir()
    (folder / "vocab.json").write_text('{"a": 0, "b": 1, "ab": 2}', encoding="utf-8")
    (folder / "merges.txt").write_text("#version: 0.2\na b\n", encoding="utf-8")
    (folder / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return folder


@pytest.fixture
def write_text_file(tmp_path: Path) -> Callable[..., Path]:
    def write(tensors: Dict[str, torch.Tensor], prefix: str = "", name: str = "qwen35.safetensors") -> Path:
        path = tmp_path / name
        save_file({prefix + k: v.detach().contiguous() for k, v in tensors.items()}, str(path))
        return path

    return write


def _load(path: Path, tokenizer_dir: Path, **options) -> MOSSImageTextEncoder:
    return load_moss_image_text_encoder(path, CPU, tokenizer_path=tokenizer_dir, **options)


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "model.",
        "language_model.",
        "language_model.model.",
        "model.language_model.",
        "text_encoders.qwen35_9b.transformer.",
        "text_encoders.qwen35_9b.transformer.model.",
    ],
)
def test_every_known_key_prefix_loads_the_same_encoder(
    write_text_file: Callable[..., Path], tokenizer_dir: Path, prefix: str
) -> None:
    source = tiny_text_model()
    path = write_text_file(source.state_dict(), prefix)

    encoder = _load(path, tokenizer_dir, dtype=torch.float32)

    with torch.no_grad():
        torch.testing.assert_close(encoder.model(IDS), source(IDS), atol=1e-5, rtol=1e-5)
    assert encoder.model.config == TINY


def test_vision_head_and_mtp_tensors_are_left_out(write_text_file: Callable[..., Path], tokenizer_dir: Path) -> None:
    extra = {
        "visual.blocks.0.weight": torch.zeros(1),
        "lm_head.weight": torch.zeros(1),
        "mtp.fc.weight": torch.zeros(1),
    }
    path = write_text_file({**tiny_text_model().state_dict(), **extra}, "model.")

    state = _load(path, tokenizer_dir).model.state_dict()

    assert not any(k.startswith(("visual.", "lm_head.", "mtp.")) for k in state)


def test_the_encoder_is_frozen_eval_stored_in_dtype_and_computes_in_fp32(
    write_text_file: Callable[..., Path], tokenizer_dir: Path
) -> None:
    source = tiny_text_model()

    encoder = _load(write_text_file(source.state_dict()), tokenizer_dir, dtype=torch.bfloat16)

    model = encoder.model
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert not any(p.requires_grad for p in model.parameters())
    assert not model.training
    assert model.compute_dtype == torch.float32
    with torch.no_grad():
        out = model(IDS)
    assert out.dtype == torch.float32
    rounded = tiny_text_model()
    rounded.load_state_dict({k: v.bfloat16().float() for k, v in source.state_dict().items()})
    assert (out - rounded(IDS)).norm() / out.norm() < 0.05


def test_the_activation_dtype_can_be_chosen(write_text_file: Callable[..., Path], tokenizer_dir: Path) -> None:
    path = write_text_file(tiny_text_model().state_dict())

    encoder = _load(path, tokenizer_dir, dtype=torch.float32, compute_dtype=torch.bfloat16)

    assert encoder.model.compute_dtype == torch.bfloat16


def test_the_loader_returns_an_encoder_that_uses_the_tokenizer_it_was_given(
    write_text_file: Callable[..., Path], tokenizer_dir: Path
) -> None:
    encoder = _load(write_text_file(tiny_text_model().state_dict()), tokenizer_dir)

    assert type(encoder.tokenizer).__name__ == "Qwen2Tokenizer"
    assert encoder.tokenizer("ab", add_special_tokens=False).input_ids == [2]


def test_an_int8_convrot_encoder_unrotates_its_embedding_and_approximates_the_float_one(
    write_text_file: Callable[..., Path], tokenizer_dir: Path
) -> None:
    source = tiny_text_model()
    path = write_text_file(text_int8_tensors(source), name="qwen35_int8_convrot.safetensors")

    encoder = _load(path, tokenizer_dir, dtype=torch.float32)

    model = encoder.model
    assert isinstance(model.embed_tokens, Int8Embedding)
    assert model.embed_tokens.weight.dtype == torch.int8
    assert model.layers[0].mlp.gate_proj.weight.dtype == torch.int8
    lookup = model.embed_tokens(IDS)
    assert (lookup - source.embed_tokens(IDS)).norm() / source.embed_tokens(IDS).norm() < 0.03
    with torch.no_grad():
        expected, got = source(IDS), model(IDS)
    assert got.shape == (1, 5, HIDDEN)
    assert (got - expected).norm() / expected.norm() < 0.1


def test_a_missing_file_is_reported_by_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Qwen3.5-9B text encoder not found"):
        load_moss_image_text_encoder(tmp_path / "absent.safetensors", CPU)


def test_a_file_that_is_not_qwen35_is_rejected(write_text_file: Callable[..., Path], tokenizer_dir: Path) -> None:
    path = write_text_file({"layers.0.mlp.gate_proj.weight": torch.zeros(2, 2)})

    with pytest.raises(ValueError, match="Not a Qwen3.5 text encoder"):
        _load(path, tokenizer_dir)


@pytest.mark.parametrize(
    ("drop", "extra"), [("norm.weight", {}), (None, {"layers.0.mlp.bogus.weight": torch.zeros(1)})]
)
def test_missing_or_unexpected_weights_are_named_in_the_error(
    write_text_file: Callable[..., Path], tokenizer_dir: Path, drop: str | None, extra: dict
) -> None:
    tensors = {k: v for k, v in tiny_text_model().state_dict().items() if k != drop}
    path = write_text_file({**tensors, **extra})

    with pytest.raises(ValueError, match="does not look like a Qwen3.5 text encoder") as error:
        _load(path, tokenizer_dir)

    assert (drop or "bogus") in str(error.value)
