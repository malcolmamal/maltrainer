"""The INT8 ConvRot loader checks which weights are int8 from the file header, not by reading every weight."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import torch

from maltrainer.models.minimax_h3 import int8_convrot

from .test_int8_convrot_loading import CPU, _layer, _write


def test_no_tensor_is_read_more_than_once_and_the_loaded_state_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensors = {**_layer("a"), **_layer("b"), "norm.weight": torch.full((3,), 2.0)}
    path = _write(tmp_path, tensors)
    reads: Counter[str] = Counter()
    real = int8_convrot.safe_open

    class Counting:
        def __init__(self, handle: Any) -> None:
            self._handle = handle

        def __enter__(self) -> Counting:
            self._handle.__enter__()
            return self

        def __exit__(self, *exc: object) -> None:
            self._handle.__exit__(*exc)

        def get_tensor(self, key: str) -> torch.Tensor:
            reads[key] += 1
            return self._handle.get_tensor(key)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._handle, name)

    monkeypatch.setattr(int8_convrot, "safe_open", lambda *a, **k: Counting(real(*a, **k)))

    state, count = int8_convrot.load_comfy_int8_convrot_state_dict(path, device=CPU)

    assert count == 2
    assert max(reads.values()) == 1
    assert torch.equal(state["a.weight"], tensors["a.weight"])
    assert state["norm.weight"].dtype == torch.float32
