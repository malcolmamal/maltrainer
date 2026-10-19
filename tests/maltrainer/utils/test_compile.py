from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch

from maltrainer.utils import compile as compile_module
from maltrainer.utils.compile import compile_blocks, compile_unavailable_reason


def _torch_with_cache_limit(monkeypatch: pytest.MonkeyPatch, limit: int) -> SimpleNamespace:
    """torch as compile.py sees it, with a stand-in dynamo config (importing dynamo takes seconds)."""
    config = SimpleNamespace(cache_size_limit=limit)
    monkeypatch.setattr(
        compile_module, "torch", SimpleNamespace(compile=torch.compile, _dynamo=SimpleNamespace(config=config))
    )
    return config


class _Block:
    def __init__(self) -> None:
        self.modes: List[Optional[str]] = []

    def compile(self, mode: Optional[str] = None) -> None:
        self.modes.append(mode)


def test_the_cpu_can_always_compile() -> None:
    assert compile_unavailable_reason(torch.device("cpu")) is None


def test_cuda_without_triton_cannot_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compile_module.importlib.util, "find_spec", lambda _name: None)

    reason = compile_unavailable_reason(torch.device("cuda"))

    assert reason == "Triton is not installed (on Windows: pip install triton-windows)"


def test_cuda_with_triton_can_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compile_module.importlib.util, "find_spec", lambda _name: object())

    assert compile_unavailable_reason(torch.device("cuda")) is None


def test_a_torch_without_compile_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(torch, "compile")

    assert compile_unavailable_reason(torch.device("cpu")) == f"torch {torch.__version__} has no torch.compile"


def test_blocks_stay_eager_when_compiling_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(compile_module.importlib.util, "find_spec", lambda _name: None)
    blocks = [_Block(), _Block()]

    with caplog.at_level(logging.WARNING, logger="maltrainer.utils.compile"):
        compiled = compile_blocks(iter(blocks), torch.device("cuda"))

    assert compiled is False
    assert [block.modes for block in blocks] == [[], []]
    assert caplog.messages == [
        "train.compile is on but the blocks stay eager: Triton is not installed (on Windows: pip install triton-windows)"
    ]


def test_every_block_is_compiled_with_room_for_recompiles(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dynamo_config = _torch_with_cache_limit(monkeypatch, 8)
    blocks = [_Block(), _Block()]

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.compile"):
        compiled = compile_blocks(iter(blocks), torch.device("cpu"), mode="max-autotune")

    assert compiled is True
    assert [block.modes for block in blocks] == [["max-autotune"], ["max-autotune"]]
    assert dynamo_config.cache_size_limit == 64
    assert caplog.messages == ["torch.compile: 2 blocks (mode max-autotune); the first steps of each new shape compile"]


def test_a_larger_cache_limit_is_kept(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    dynamo_config = _torch_with_cache_limit(monkeypatch, 256)

    with caplog.at_level(logging.INFO, logger="maltrainer.utils.compile"):
        compile_blocks([_Block()], torch.device("cpu"))

    assert dynamo_config.cache_size_limit == 256
    assert caplog.messages == ["torch.compile: 1 blocks (mode default); the first steps of each new shape compile"]
