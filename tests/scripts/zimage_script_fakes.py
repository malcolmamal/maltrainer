"""A fake Z-Image pipeline, a tiny Qwen tokenizer and encoder, and a guard for scripts that configure logging."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import diffusers
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

WORDS = ["[UNK]", "[PAD]", "a", "photo", "of", "cat"]


class FakePipeline:
    """Stands for ``ZImagePipeline``: records what the script does to it and returns one blank image per call."""

    instances: ClassVar[list[FakePipeline]]
    fail_lora: ClassVar[bool]

    def __init__(self, model_id: str, dtype: torch.dtype) -> None:
        self.model_id, self.dtype, self.device = model_id, dtype, "unmoved"
        self.events: list[str] = []
        self.calls: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, torch_dtype: torch.dtype) -> FakePipeline:
        cls.instances.append(cls(model_id, torch_dtype))
        return cls.instances[-1]

    def to(self, device: str) -> FakePipeline:
        self.device = device
        self.events.append(f"to {device}")
        return self

    def load_lora_weights(self, path: str) -> None:
        if self.fail_lora:
            raise RuntimeError("bad lora")
        self.events.append(f"load_lora {path}")

    def fuse_lora(self) -> None:
        self.events.append("fuse")

    def __call__(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls.append({"args": args, **kwargs})
        return SimpleNamespace(images=[Image.new("RGB", (4, 4), (len(self.calls), 0, 0))])


def pipeline_class(fail_lora: bool = False) -> type[FakePipeline]:
    """A fresh fake pipeline class, so each test sees only the pipelines its own script run built."""
    return type("Pipeline", (FakePipeline,), {"instances": [], "fail_lora": fail_lora})


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")


def tiny_text_encoder() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=len(WORDS), hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=4, tie_word_embeddings=False,
    )  # fmt: skip
    with torch.random.fork_rng():
        torch.manual_seed(4)
        return Qwen3ForCausalLM(config)


class Loader:
    """Stands where a ``from_pretrained`` class stood in the script; ``asked`` records each request."""

    def __init__(self, build: Any, asked: list[tuple[str, str]]) -> None:
        self.build, self.asked = build, asked

    def from_pretrained(self, model_id: str, subfolder: str, torch_dtype: torch.dtype | None = None) -> Any:
        self.asked.append((model_id, subfolder))
        model = self.build()
        return model.to(torch_dtype) if torch_dtype is not None else model


@contextmanager
def restored_logging() -> Iterator[None]:
    """Scripts call ``logging.basicConfig`` when they load; undo its handlers and level afterwards."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        yield
    finally:
        for handler in [h for h in root.handlers if h not in handlers]:
            root.removeHandler(handler)
        root.setLevel(level)


@contextmanager
def loaded_infer(monkeypatch: pytest.MonkeyPatch, pipeline: type[FakePipeline]) -> Iterator[ModuleType]:
    """``infer_zimage`` imported while ``diffusers.ZImagePipeline`` is the fake (so the real one never loads)."""
    monkeypatch.setitem(vars(diffusers), "ZImagePipeline", pipeline)
    with restored_logging():
        module = importlib.import_module("infer_zimage")
    try:
        yield module
    finally:
        sys.modules.pop("infer_zimage", None)
