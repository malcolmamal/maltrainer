from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from maltrainer.data import disk_cache
from maltrainer.data.disk_cache import model_identity


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("path", [None, ""])
def test_no_model_path_has_the_identity_none(path: str | None) -> None:
    assert model_identity(path) == "none"


def test_a_small_model_is_identified_by_name_size_and_all_its_bytes(tmp_path: Path) -> None:
    path = _write(tmp_path / "vae.safetensors", b"12345")

    assert model_identity(path) == f"vae.safetensors|5|{hashlib.blake2b(b'12345', digest_size=8).hexdigest()}"


def test_the_same_file_in_another_folder_has_the_same_identity(tmp_path: Path) -> None:
    here = _write(tmp_path / "here" / "vae.safetensors", b"12345")
    there = _write(tmp_path / "there" / "vae.safetensors", b"12345")

    assert model_identity(str(there)) == model_identity(here)


def test_a_model_of_another_size_has_another_identity(tmp_path: Path) -> None:
    path = _write(tmp_path / "vae.safetensors", b"12345")
    before = model_identity(path)

    _write(path, b"123456")

    assert model_identity(path) != before


@pytest.mark.parametrize("changed", [0, 99])
def test_two_models_of_one_name_and_size_differ_by_their_first_or_last_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: int
) -> None:
    monkeypatch.setattr(disk_cache, "IDENTITY_SAMPLE", 10)
    data = bytearray(range(100))
    base = _write(tmp_path / "a" / "te.safetensors", bytes(data))
    data[changed] ^= 0xFF
    tuned = _write(tmp_path / "b" / "te.safetensors", bytes(data))

    assert model_identity(base) != model_identity(tuned)


def test_only_the_first_and_last_sample_are_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disk_cache, "IDENTITY_SAMPLE", 10)
    data = bytearray(range(100))
    base = _write(tmp_path / "a" / "te.safetensors", bytes(data))
    data[50] ^= 0xFF
    middle_changed = _write(tmp_path / "b" / "te.safetensors", bytes(data))

    assert model_identity(base) == model_identity(middle_changed)


@pytest.mark.parametrize("size", [10, 15, 20])
def test_a_file_up_to_two_samples_long_is_hashed_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    monkeypatch.setattr(disk_cache, "IDENTITY_SAMPLE", 10)
    data = bytes(range(size))
    path = _write(tmp_path / "te.safetensors", data)

    assert model_identity(path) == f"te.safetensors|{size}|{hashlib.blake2b(data, digest_size=8).hexdigest()}"


def test_a_missing_model_is_identified_by_its_name_alone(tmp_path: Path) -> None:
    assert model_identity(tmp_path / "missing.safetensors") == "missing.safetensors"
