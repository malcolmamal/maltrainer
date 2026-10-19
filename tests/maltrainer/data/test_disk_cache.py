from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.data.disk_cache import CACHE_DIR_NAME, FolderCache, cache_key, content_hash


def test_the_same_bytes_hash_the_same_wherever_the_file_is(tmp_path: Path) -> None:
    first = tmp_path / "a" / "ben.png"
    second = tmp_path / "b" / "renamed.png"
    for path in (first, second):
        path.parent.mkdir()
        path.write_bytes(b"pixels")

    assert content_hash(first) == content_hash(second)
    assert content_hash(str(first)) == content_hash(second)


def test_changing_one_byte_changes_the_content_hash(tmp_path: Path) -> None:
    path = tmp_path / "ben.png"
    path.write_bytes(b"pixels")
    before = content_hash(path)

    path.write_bytes(b"pixelz")

    assert content_hash(path) != before


def test_a_file_larger_than_one_read_block_hashes_all_its_bytes(tmp_path: Path) -> None:
    data = bytes(range(256)) * 4097
    path = tmp_path / "big.bin"
    path.write_bytes(data)

    assert content_hash(path) == hashlib.blake2b(data, digest_size=20).hexdigest()
    assert len(content_hash(path)) == 40


def test_a_cache_key_is_a_stable_sha1_of_its_parts() -> None:
    key = cache_key("latent/v1", "abc", 512, "bucket")

    assert key == cache_key("latent/v1", "abc", 512, "bucket")
    assert key == hashlib.sha1("latent/v1\x1fabc\x1f512\x1fbucket".encode()).hexdigest()


@pytest.mark.parametrize(
    "other",
    [("latent/v1", "abd", 512, "bucket"), ("latent/v1", "abc", 1024, "bucket"), ("latent/v1", "abc", 512, "square")],
)
def test_changing_any_part_changes_the_cache_key(other: tuple) -> None:
    assert cache_key(*other) != cache_key("latent/v1", "abc", 512, "bucket")


def test_part_boundaries_and_order_are_part_of_the_cache_key() -> None:
    assert cache_key("ab", "c") != cache_key("a", "bc")
    assert cache_key("a", "b") != cache_key("b", "a")


def test_a_folder_cache_lives_in_the_image_folder(tmp_path: Path) -> None:
    assert FolderCache(tmp_path).root == tmp_path / CACHE_DIR_NAME
    assert FolderCache(str(tmp_path)).root == tmp_path / CACHE_DIR_NAME


def test_a_stored_tensor_reads_back_with_its_values_and_dtype(tmp_path: Path) -> None:
    cache = FolderCache(tmp_path)
    tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)

    cache.put("latents", "k1", tensor)

    assert cache.has("latents", "k1")
    loaded = cache.get("latents", "k1")
    assert loaded is not None
    assert loaded.dtype == torch.bfloat16
    assert torch.equal(loaded, tensor)


def test_an_entry_is_one_safetensors_file_per_kind_and_key_with_no_temp_file_left(tmp_path: Path) -> None:
    cache = FolderCache(tmp_path)

    cache.put("text", "k1", torch.ones(2))

    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        f"{CACHE_DIR_NAME}/text/k1.safetensors"
    ]


def test_a_non_contiguous_tensor_that_needs_grad_is_stored_detached(tmp_path: Path) -> None:
    cache = FolderCache(tmp_path)
    tensor = torch.arange(6.0).reshape(2, 3).t().requires_grad_()

    cache.put("latents", "k1", tensor)

    loaded = cache.get("latents", "k1")
    assert loaded is not None
    assert not loaded.requires_grad
    assert torch.equal(loaded, tensor.detach())


def test_putting_the_same_key_again_replaces_the_entry(tmp_path: Path) -> None:
    cache = FolderCache(tmp_path)
    cache.put("latents", "k1", torch.zeros(2))

    cache.put("latents", "k1", torch.ones(3))

    assert torch.equal(cache.get("latents", "k1"), torch.ones(3))


def test_an_unknown_key_or_another_kind_is_a_miss(tmp_path: Path) -> None:
    cache = FolderCache(tmp_path)
    cache.put("latents", "k1", torch.ones(2))

    assert not cache.has("latents", "k2")
    assert cache.get("latents", "k2") is None
    assert not cache.has("text", "k1")
    assert cache.get("text", "k1") is None


def test_a_truncated_entry_is_a_logged_miss(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    cache = FolderCache(tmp_path)
    cache.put("latents", "k1", torch.ones(64))
    path = tmp_path / CACHE_DIR_NAME / "latents" / "k1.safetensors"
    path.write_bytes(path.read_bytes()[:20])

    with caplog.at_level(logging.WARNING, logger="maltrainer.data.disk_cache"):
        loaded = cache.get("latents", "k1")

    assert loaded is None
    assert cache.has("latents", "k1")
    assert "Unreadable cache entry" in caplog.text


def test_a_file_without_the_tensor_entry_is_a_miss(tmp_path: Path) -> None:
    cache = FolderCache(tmp_path)
    path = tmp_path / CACHE_DIR_NAME / "latents" / "k1.safetensors"
    path.parent.mkdir(parents=True)
    save_file({"other": torch.ones(2)}, str(path))

    assert cache.get("latents", "k1") is None
