from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from maltrainer.models.minimax_h3.text_encoder import MiniMaxH3TextEncoder

DIM = 4


class FakeTokenizer:
    """Whitespace tokenizer: word i of a prompt gets id 10 + len(word); rows are right-padded with 0."""

    pad_token: str | None = "<pad>"
    pad_token_id: int | None = 0

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.padding_side = "left"

    def __call__(self, prompts: list[str], **options: object) -> SimpleNamespace:
        self.calls.append({"prompts": prompts, "padding_side": self.padding_side, **options})
        rows = [[10 + len(word) for word in prompt.split()] for prompt in prompts]
        width = max(len(row) for row in rows)
        ids = torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long).view(len(rows), width)
        mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]).view(len(rows), width)
        return SimpleNamespace(input_ids=ids, attention_mask=mask)


class FakeLanguageModel(nn.Module):
    def __init__(self, extra: int = 0, as_tuple: bool = False) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor, object]] = []
        self.extra, self.as_tuple = extra, as_tuple

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: object) -> object:
        self.calls.append((input_ids, attention_mask, use_cache))
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, DIM)
        hidden = torch.cat([hidden, hidden[:, :1].expand(-1, self.extra, -1)], dim=1)
        return (hidden,) if self.as_tuple else SimpleNamespace(last_hidden_state=hidden)


def encoder(language_model: nn.Module | None = None) -> MiniMaxH3TextEncoder:
    wrapper = nn.Module()
    wrapper.language_model = language_model or FakeLanguageModel()
    return MiniMaxH3TextEncoder(wrapper, FakeTokenizer())
