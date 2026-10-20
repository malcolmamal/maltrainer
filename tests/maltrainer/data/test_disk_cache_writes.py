"""The cache is optional: a write that cannot happen is a warning, never the end of a run."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors import SafetensorError

from maltrainer.data import disk_cache
from maltrainer.data.disk_cache import CACHE_DIR_NAME, FolderCache


def _leftovers(root: Path) -> list[Path]:
    return [path for path in (root / CACHE_DIR_NAME).rglob("*") if path.is_file()]


@pytest.mark.parametrize("error", [OSError("disk full"), SafetensorError("write failed")])
def test_a_failed_write_is_skipped_with_a_warning_and_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, error: Exception
) -> None:
    def failing_save(_tensors: object, path: str) -> None:
        Path(path).write_bytes(b"partial")
        raise error

    monkeypatch.setattr(disk_cache, "save_file", failing_save)
    cache = FolderCache(tmp_path)

    with caplog.at_level(logging.WARNING, logger="maltrainer.data.disk_cache"):
        cache.put("latents", "k1", torch.ones(1))

    assert not cache.has("latents", "k1")
    assert _leftovers(tmp_path) == []
    assert "Not caching latents/k1" in caplog.text


def test_a_folder_that_cannot_be_written_to_is_skipped_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def refused(**_options: object) -> tuple[int, str]:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(tempfile, "mkstemp", refused)

    with caplog.at_level(logging.WARNING, logger="maltrainer.data.disk_cache"):
        FolderCache(tmp_path).put("text", "k2", torch.ones(1))

    assert "Not caching text/k2" in caplog.text
    assert _leftovers(tmp_path) == []


def test_an_error_that_is_not_a_storage_error_still_propagates_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_save(_tensors: object, path: str) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("a bug, not a full disk")

    monkeypatch.setattr(disk_cache, "save_file", broken_save)

    with pytest.raises(RuntimeError, match="a bug"):
        FolderCache(tmp_path).put("latents", "k3", torch.ones(1))

    assert _leftovers(tmp_path) == []


def test_a_good_write_after_a_skipped_one_is_stored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = FolderCache(tmp_path)
    real_save = disk_cache.save_file
    monkeypatch.setattr(disk_cache, "save_file", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
    cache.put("latents", "k4", torch.ones(1))
    monkeypatch.setattr(disk_cache, "save_file", real_save)

    cache.put("latents", "k4", torch.full((1,), 2.0))

    torch.testing.assert_close(cache.get("latents", "k4"), torch.full((1,), 2.0))
