from __future__ import annotations

import pytest
import torch
from torch import nn

from maltrainer.models.embedding import TextualInversionEmbedding

VOCAB = ["<pad>", "a", "photo", "of", "dog"]


class _Tokenizer:
    def __init__(self) -> None:
        self.vocab = {token: index for index, token in enumerate(VOCAB)}

    def __len__(self) -> int:
        return len(self.vocab)

    def add_tokens(self, tokens: list[str]) -> int:
        new = [token for token in tokens if token not in self.vocab]
        for token in new:
            self.vocab[token] = len(self.vocab)
        return len(new)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.vocab[token] for token in tokens]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.vocab[word] for word in text.split()]
        return ids if not add_special_tokens else [0, *ids]


class _TextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), 3)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(len(VOCAB) * 3, dtype=torch.float32).view(-1, 3))

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def resize_token_embeddings(self, size: int) -> None:
        grown = nn.Embedding(size, 3)
        with torch.no_grad():
            grown.weight.fill_(-1.0)
            grown.weight[: self.embed.num_embeddings] = self.embed.weight
        self.embed = grown


def _row(word: str) -> torch.Tensor:
    index = VOCAB.index(word)
    return torch.arange(index * 3, index * 3 + 3, dtype=torch.float32)


def _embedding(**options: object) -> tuple[_Tokenizer, _TextEncoder, TextualInversionEmbedding]:
    tokenizer, encoder = _Tokenizer(), _TextEncoder()
    return tokenizer, encoder, TextualInversionEmbedding(tokenizer, encoder, **options)


def test_the_trigger_token_is_added_and_the_table_grows_to_match() -> None:
    tokenizer, encoder, embedding = _embedding(trigger_token="zwx")

    assert embedding.token_ids == [len(VOCAB)]
    assert tokenizer.vocab["zwx"] == len(VOCAB)
    assert encoder.get_input_embeddings().num_embeddings == len(VOCAB) + 1


def test_the_new_vector_starts_as_the_first_init_text_token() -> None:
    _, _, embedding = _embedding(init_text="dog photo")

    torch.testing.assert_close(embedding.get_embeddings(), _row("dog").unsqueeze(0))


def test_extra_vectors_get_numbered_tokens_and_reuse_the_last_init_token() -> None:
    tokenizer, _, embedding = _embedding(trigger_token="sks", num_vectors=3, init_text="a photo")

    assert tokenizer.convert_tokens_to_ids(["sks", "sks_1", "sks_2"]) == embedding.token_ids
    torch.testing.assert_close(embedding.get_embeddings(), torch.stack([_row("a"), _row("photo"), _row("photo")]))


def test_a_trigger_already_in_the_vocabulary_is_rejected() -> None:
    with pytest.raises(ValueError, match="Token 'dog' may already exist"):
        _embedding(trigger_token="dog")


def test_the_state_dict_holds_the_learned_vectors_under_embedding() -> None:
    _, _, embedding = _embedding(num_vectors=2, init_text="of dog")

    state = embedding.get_state_dict()

    assert list(state) == ["embedding"]
    torch.testing.assert_close(state["embedding"], torch.stack([_row("of"), _row("dog")]))


def test_loading_writes_the_vectors_into_only_the_trigger_rows() -> None:
    _, encoder, embedding = _embedding(num_vectors=2)
    vectors = torch.tensor([[1.0, 2.0, 3.0], [-4.0, -5.0, -6.0]])

    embedding.load_state_dict({"embedding": vectors})

    table = encoder.get_input_embeddings().weight.detach()
    torch.testing.assert_close(table[embedding.token_ids], vectors)
    torch.testing.assert_close(table[: len(VOCAB)], torch.arange(len(VOCAB) * 3, dtype=torch.float32).view(-1, 3))


def test_an_optimizer_takes_the_parameters_and_moves_the_trigger_vector() -> None:
    _, encoder, embedding = _embedding(trigger_token="sks", init_text="dog")
    optimizer = torch.optim.SGD(embedding.parameters(), lr=1.0)
    trigger = embedding.token_ids[0]

    embedding(torch.tensor([trigger])).sum().backward()
    optimizer.step()

    assert list(embedding.parameters()) == [encoder.get_input_embeddings().weight]
    torch.testing.assert_close(embedding.get_embeddings()[0], _row("dog") - 1.0)


def test_forward_looks_up_the_vectors_of_its_token_ids() -> None:
    _, _, embedding = _embedding(trigger_token="sks", init_text="photo")

    vectors = embedding(torch.tensor([VOCAB.index("a"), embedding.token_ids[0]]))

    torch.testing.assert_close(vectors, torch.stack([_row("a"), _row("photo")]))
