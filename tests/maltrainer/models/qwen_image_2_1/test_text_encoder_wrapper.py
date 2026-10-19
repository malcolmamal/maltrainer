from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maltrainer.models.minimax_h3.int8_convrot import rotate_activation
from maltrainer.models.qwen_image_2_1.text_encoder import (
    DEFAULT_DROP_IDX,
    DEFAULT_SYSTEM_PROMPT,
    Int8Embedding,
    Qwen3VLTextEncoder,
    _map_te_key,
)


class _Tokenizer:
    def __init__(self, length: int = 20) -> None:
        self.length = length
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, texts: list[str], **options: object) -> SimpleNamespace:
        self.calls.append((texts, options))
        ids = torch.arange(1, self.length + 1).repeat(len(texts), 1)
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


class _LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.seen: dict[str, torch.Tensor] = {}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        self.seen = {"input_ids": input_ids, "attention_mask": attention_mask}
        return SimpleNamespace(last_hidden_state=input_ids.float()[..., None].expand(-1, -1, 3) * self.weight)


def _encoder(length: int = 20, **options: object) -> Qwen3VLTextEncoder:
    model = nn.Module()
    model.language_model = _LanguageModel()
    return Qwen3VLTextEncoder(model, _Tokenizer(length), **options)


def test_the_prompt_is_wrapped_in_the_chat_template_with_the_system_prompt() -> None:
    encoder = _encoder()

    text = encoder.format_prompt("a cat")

    assert text == (
        f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
    )


def test_a_custom_system_prompt_replaces_the_default() -> None:
    assert "<|im_start|>system\nBe brief.<|im_end|>" in _encoder(system_prompt="Be brief.").format_prompt("x")


def test_a_single_prompt_is_tokenised_as_a_batch_of_one_with_the_documented_options() -> None:
    encoder = _encoder()

    encoder.encode_prompts("a cat", max_length=77)

    texts, options = encoder.tokenizer.calls[0]
    assert texts == [encoder.format_prompt("a cat")]
    assert options == {
        "padding": "longest",
        "max_length": 77,
        "truncation": True,
        "add_special_tokens": False,
        "return_tensors": "pt",
    }


def test_the_system_prompt_tokens_are_dropped_from_the_returned_hidden_states() -> None:
    encoder = _encoder(length=20)

    out = encoder.encode_prompts(["a", "b"])

    assert DEFAULT_DROP_IDX == 14
    assert out.shape == (2, 6, 3)
    assert out[0, :, 0].tolist() == [15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    assert encoder.model.language_model.seen["input_ids"].shape == (2, 20)


@pytest.mark.parametrize(("length", "drop_idx", "expected"), [(14, 14, 14), (10, 14, 10), (20, 0, 20), (20, 5, 15)])
def test_nothing_is_dropped_unless_the_sequence_is_longer_than_the_drop_index(
    length: int, drop_idx: int, expected: int
) -> None:
    out = _encoder(length=length, drop_idx=drop_idx).encode_prompts("a")

    assert out.shape[1] == expected


def test_encoding_runs_without_gradients() -> None:
    out = _encoder().encode_prompts("a")

    assert not out.requires_grad


def test_tensors_go_to_the_requested_device_or_else_to_the_models_device() -> None:
    explicit = _encoder().encode_prompts("a", device=torch.device("cpu"))
    default = _encoder().encode_prompts("a")

    assert explicit.device.type == default.device.type == "cpu"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("model.visual.blocks.0.attn.qkv.weight", "visual.blocks.0.attn.qkv.weight"),
        ("model.layers.3.mlp.up_proj.weight", "language_model.layers.3.mlp.up_proj.weight"),
        ("model.embed_tokens.weight", "language_model.embed_tokens.weight"),
        ("lm_head.weight", "lm_head.weight"),
        ("visual.merger.weight", "visual.merger.weight"),
    ],
)
def test_checkpoint_keys_are_mapped_onto_the_visual_and_language_model_namespaces(key: str, expected: str) -> None:
    assert _map_te_key(key) == expected


def test_an_embedding_without_rotation_is_the_int8_row_times_its_scale_in_bfloat16() -> None:
    table = Int8Embedding(num_embeddings=3, embedding_dim=4)
    with torch.no_grad():
        table.weight.copy_(torch.tensor([[1, 2, 3, 4], [-4, 0, 4, 8], [0, 0, 0, 0]], dtype=torch.int8))
        table.scale_weight.copy_(torch.tensor([[0.5], [0.25], [1.0]]))

    out = table(torch.tensor([[1, 0]]))

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), torch.tensor([[[-1.0, 0.0, 1.0, 2.0], [0.5, 1.0, 1.5, 2.0]]]))


def test_an_embedding_with_a_group_size_undoes_the_hadamard_rotation_of_the_stored_table() -> None:
    table = Int8Embedding(num_embeddings=2, embedding_dim=4, group_size=4)
    with torch.no_grad():
        table.weight.copy_(torch.tensor([[2, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.int8))
        table.scale_weight.copy_(torch.tensor([[1.0], [2.0]]))

    out = table(torch.tensor([0, 1]))

    torch.testing.assert_close(out[0].float(), torch.tensor([1.0, 1.0, 1.0, -1.0]))
    torch.testing.assert_close(out[1].float(), rotate_activation(torch.full((1, 4), 2.0), 4)[0])
    assert table.int8_convrot_groupsize.dtype == torch.int32


def test_the_embedding_table_is_a_frozen_int8_parameter_with_buffers_for_scale_and_group() -> None:
    table = Int8Embedding(5, 8, group_size=4)

    assert table.weight.dtype == torch.int8
    assert not table.weight.requires_grad
    assert table.scale_weight.shape == (5, 1)
    assert sorted(table.state_dict()) == ["int8_convrot_groupsize", "scale_weight", "weight"]
