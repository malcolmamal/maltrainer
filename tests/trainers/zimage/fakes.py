"""Stand-ins for what the Z-Image trainers fetch from the hub: each ``from_pretrained`` builds a tiny real model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
import torch
from diffusers.models.transformers import ZImageTransformer2DModel

from .tiny import build_text_encoder, build_tokenizer, build_transformer, build_vae


@dataclass
class Call:
    """One transformer forward: what the trainer fed it."""

    x: torch.Tensor
    t: torch.Tensor
    cap_feats: torch.Tensor
    cap_mask: torch.Tensor | None = None


@dataclass
class Loaded:
    kind: str
    path: str
    subfolder: str | None
    dtype: torch.dtype | None


@dataclass
class Fakes:
    """What the trainer asked for (``loads``, ``downloads``) and fed the models (``calls``, ``prompts``).

    ``output`` shapes what the transformer returns: "list" (diffusers' own: a list of [C, 1, H, W]), "tuple",
    "tensor" ([B, C, 1, H, W]), "plain" ([B, C, H, W]) or "swapped" ([C, B, H, W]).
    """

    output: str = "list"
    masked: bool = False
    chat: str = "template"
    loads: list[Loaded] = field(default_factory=list)
    downloads: list[tuple[str, str]] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    token_ids: list[torch.Tensor] = field(default_factory=list)
    parts: dict[str, Any] = field(default_factory=dict)

    def prompts(self) -> list[str]:
        """The text-encoder inputs, decoded: one string per sample per call."""
        tokenizer = self.parts["tokenizer"]
        return [tokenizer.decode(row) for ids in self.token_ids for row in ids]


def _shaped(out: Any, mode: str) -> Any:
    if mode == "list":
        return out
    stacked = torch.stack(list(out.sample))
    if mode == "tuple":
        return (stacked,)
    return {
        "tensor": stacked,
        "plain": stacked.squeeze(2),
        "swapped": stacked.squeeze(2).permute(1, 0, 2, 3),
    }[mode]


def make_transformer(fakes: Fakes) -> ZImageTransformer2DModel:
    """The tiny transformer with a ``forward`` that records its inputs and returns ``fakes.output``'s shape."""
    model = build_transformer()
    real = model.forward

    def forward(x: Any, t: Any, cap_feats: Any, cap_mask: torch.Tensor | None = None) -> Any:
        fakes.calls.append(Call(x.detach().clone(), t.detach().clone(), cap_feats.detach().clone(), cap_mask))
        return _shaped(real(x, t, cap_feats), fakes.output)

    def plain(x: Any, t: Any, cap_feats: Any) -> Any:
        return forward(x, t, cap_feats)

    def masked(x: Any, t: Any, cap_feats: Any, cap_mask: torch.Tensor) -> Any:
        return forward(x, t, cap_feats, cap_mask)

    cast(Any, model).forward = masked if fakes.masked else plain
    return model


class TokenizerWithoutTemplate:
    """The tokenizer with no ``apply_chat_template`` at all (``chat="none"``) or one without keyword arguments."""

    def __init__(self, inner: Any, chat: str) -> None:
        self.inner = inner
        if chat == "plain":
            self.apply_chat_template = self._plain

    def _plain(self, conversation: list[dict[str, str]], tokenize: bool, add_generation_prompt: bool) -> str:
        return cast(
            str,
            self.inner.apply_chat_template(
                conversation, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            ),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner(*args, **kwargs)

    def __len__(self) -> int:
        return len(self.inner)

    def __getattr__(self, name: str) -> Any:
        if name == "apply_chat_template":
            raise AttributeError(name)
        return getattr(self.inner, name)


class Loader:
    """Stands where a ``from_pretrained`` class stood: records the request, returns ``build()``'s model."""

    def __init__(self, kind: str, fakes: Fakes, build: Callable[[], Any]) -> None:
        self.kind, self.fakes, self.build = kind, fakes, build

    def from_pretrained(self, path: str, subfolder: str | None = None, torch_dtype: torch.dtype | None = None) -> Any:
        model = self.build()
        if isinstance(model, torch.nn.Module):
            model = model.to(torch_dtype).eval()
        self.fakes.loads.append(Loaded(self.kind, path, subfolder, torch_dtype))
        self.fakes.parts[self.kind] = model
        return model


def _text_encoder(fakes: Fakes) -> Any:
    model = build_text_encoder()

    def note(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        fakes.token_ids.append(args[0].detach().clone())

    model.model.register_forward_pre_hook(note)
    return model


def _tokenizer(fakes: Fakes) -> Any:
    tokenizer = build_tokenizer()
    return tokenizer if fakes.chat == "template" else TokenizerWithoutTemplate(tokenizer, fakes.chat)


def install_fakes(monkeypatch: pytest.MonkeyPatch, lora: object, embedding: object) -> Fakes:
    """Put the loaders where the shared model module imports them (``lora``), and the embedding trainer's own
    text encoder class; ``fakes`` is shared by both."""
    fakes = Fakes()
    loaders = {
        "transformer": Loader("transformer", fakes, lambda: make_transformer(fakes)),
        "text_encoder": Loader("text_encoder", fakes, lambda: _text_encoder(fakes)),
        "tokenizer": Loader("tokenizer", fakes, lambda: _tokenizer(fakes)),
        "vae": Loader("vae", fakes, build_vae),
    }

    def download(repo_id: str, filename: str) -> str:
        fakes.downloads.append((repo_id, filename))
        return filename

    monkeypatch.setattr(lora, "ZImageTransformer2DModel", loaders["transformer"])
    monkeypatch.setattr(lora, "AutoTokenizer", loaders["tokenizer"])
    monkeypatch.setattr(lora, "AutoencoderKL", loaders["vae"])
    monkeypatch.setattr(lora, "AutoModelForCausalLM", loaders["text_encoder"])
    monkeypatch.setattr(embedding, "Qwen3ForCausalLM", loaders["text_encoder"])
    monkeypatch.setattr(lora, "hf_hub_download", download)
    return fakes
