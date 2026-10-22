from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maltrainer.models.minimax_h3.int8_convrot import rotate_activation
from maltrainer.models.qwen_image_2_1 import text_encoder as te
from maltrainer.models.qwen_image_2_1.text_encoder import Int8Embedding, Qwen3VLTextEncoder, _map_te_key


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask})
        hidden = input_ids.unsqueeze(-1).float().expand(-1, -1, 3).clone()
        return SimpleNamespace(last_hidden_state=hidden)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.language_model = FakeLanguageModel()


class FakeTokenizer:
    def __init__(self, length: int) -> None:
        self.length = length
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, texts: list[str], **kwargs: object) -> SimpleNamespace:
        self.calls.append((texts, kwargs))
        ids = torch.arange(self.length).repeat(len(texts), 1)
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_the_embedding_dequantizes_rows_by_their_own_scale_without_rotation_at_group_size_zero() -> None:
    emb = Int8Embedding(3, 4, group_size=0)
    emb.weight.data = torch.tensor([[1, 2, 3, 4], [-5, 0, 5, 10], [7, 7, 7, 7]], dtype=torch.int8)
    emb.scale_weight.copy_(torch.tensor([[0.5], [2.0], [0.1]]))

    out = emb(torch.tensor([[1, 0]]))

    assert out.dtype == torch.bfloat16 and out.shape == (1, 2, 4)
    assert torch.equal(out[0, 0], torch.tensor([-10.0, 0.0, 10.0, 20.0], dtype=torch.bfloat16))
    assert torch.equal(out[0, 1], torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16))


def test_the_embedding_applies_the_hadamard_rotation_per_group_of_the_stored_row() -> None:
    emb = Int8Embedding(1, 8, group_size=4)
    emb.weight.data = torch.tensor([[2, 0, 0, 0, 0, 4, 0, 0]], dtype=torch.int8)

    out = emb(torch.tensor([[0]]))[0, 0].float()

    # x @ H picks row i of H scaled by x[i]: H4 rows are (1,1,1,-1), (1,1,-1,1), ... over 2
    assert torch.allclose(out, torch.tensor([1.0, 1.0, 1.0, -1.0, 2.0, 2.0, -2.0, 2.0]))


def test_rotating_an_embedding_row_twice_returns_it_because_the_hadamard_is_an_involution() -> None:
    row = torch.randn(1, 16)

    assert torch.allclose(rotate_activation(rotate_activation(row, 4), 4), row, atol=1e-5)


def test_the_embedding_records_its_group_size_in_a_buffer_and_has_no_trainable_weight() -> None:
    emb = Int8Embedding(5, 8, group_size=4)

    assert emb.int8_convrot_groupsize.item() == 4 and emb.int8_convrot_groupsize.dtype == torch.int32
    assert emb.weight.dtype == torch.int8 and not emb.weight.requires_grad
    assert emb.scale_weight.shape == (5, 1)


@pytest.mark.parametrize(
    ("key", "mapped"),
    [
        ("model.visual.blocks.0.attn.weight", "visual.blocks.0.attn.weight"),
        ("model.layers.3.mlp.weight", "language_model.layers.3.mlp.weight"),
        ("model.embed_tokens.weight", "language_model.embed_tokens.weight"),
        ("lm_head.weight", "lm_head.weight"),
    ],
)
def test_checkpoint_keys_map_onto_the_vl_model_namespaces(key: str, mapped: str) -> None:
    assert _map_te_key(key) == mapped


def test_the_prompt_is_wrapped_in_the_chat_template_with_the_system_prompt() -> None:
    encoder = Qwen3VLTextEncoder(FakeModel(), FakeTokenizer(20), system_prompt="Be brief.")

    assert encoder.format_prompt("a cat") == (
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n"
    )
    assert te.DEFAULT_DROP_IDX == 14


def test_encoding_drops_the_leading_template_tokens_and_keeps_the_rest() -> None:
    model, tokenizer = FakeModel(), FakeTokenizer(20)
    encoder = Qwen3VLTextEncoder(model, tokenizer)

    out = encoder.encode_prompts("a cat")

    assert out.shape == (1, 6, 3)
    assert out[0, :, 0].tolist() == [14.0, 15.0, 16.0, 17.0, 18.0, 19.0]


def test_encoding_batches_a_list_and_forwards_tokenizer_options_and_the_mask() -> None:
    model, tokenizer = FakeModel(), FakeTokenizer(20)
    encoder = Qwen3VLTextEncoder(model, tokenizer, drop_idx=4)

    out = encoder.encode_prompts(["a", "b"], max_length=77)

    texts, kwargs = tokenizer.calls[0]
    assert len(texts) == 2 and texts[0] == encoder.format_prompt("a")
    assert kwargs["max_length"] == 77 and kwargs["truncation"] is True and kwargs["padding"] == "longest"
    assert kwargs["add_special_tokens"] is False
    assert out.shape == (2, 16, 3)
    assert model.language_model.calls[0]["attention_mask"].shape == (2, 20)


def test_sequences_no_longer_than_drop_idx_are_returned_untouched() -> None:
    encoder = Qwen3VLTextEncoder(FakeModel(), FakeTokenizer(14))

    assert encoder.encode_prompts("x").shape == (1, 14, 3)


def test_a_zero_drop_idx_keeps_every_token() -> None:
    encoder = Qwen3VLTextEncoder(FakeModel(), FakeTokenizer(20), drop_idx=0)

    assert encoder.encode_prompts("x").shape == (1, 20, 3)


def test_an_explicit_device_is_used_for_the_token_tensors() -> None:
    model = FakeModel()
    encoder = Qwen3VLTextEncoder(model, FakeTokenizer(20))

    encoder.encode_prompts("x", device=torch.device("cpu"))

    assert model.language_model.calls[0]["input_ids"].device.type == "cpu"
