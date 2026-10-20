"""Encoding the dataset once: the disk cache next to the images, its keys, and what a miss re-encodes."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

import train_moss_image_lora as lora
from maltrainer.data.disk_cache import CACHE_DIR_NAME

from .fakes import Fakes
from .runs import cached_lora
from .world import Workspace, write_image


def _cached(ws: Workspace, **dataset: Any) -> lora.MOSSImageLoRATrainer:
    return cached_lora(ws.lora_config(dataset={"cache": True, **dataset}))


def _entries(folder: Path, kind: str) -> list[Path]:
    return sorted((folder / CACHE_DIR_NAME / kind).glob("*.safetensors"))


def _with_model_files(ws: Workspace) -> None:
    (ws.root / "vae.safetensors").write_bytes(b"vae-1")
    (ws.root / "text.safetensors").write_bytes(b"text-1")


def test_a_first_run_encodes_and_writes_one_entry_per_image_and_caption_beside_the_images(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images("p", 3, caption=None)

    trainer = _cached(workspace)

    assert (fakes.vae_loads, fakes.encoder.encoded, len(fakes.text_loads)) == (1, 3, 1)
    assert len(_entries(workspace.data / "p", "latents")) == 3
    assert len(_entries(workspace.data / "p", "text")) >= 1
    assert sorted(trainer.latent_cache) == sorted(str(path) for path in (workspace.data / "p").glob("*.png"))


def test_a_second_run_reads_everything_from_the_cache_and_loads_no_encoder(
    workspace: Workspace, fakes: Fakes, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 3)
    first = _cached(workspace)

    with caplog.at_level(logging.INFO):
        second = _cached(workspace)

    assert (fakes.vae_loads, len(fakes.text_loads)) == (1, 1)
    assert "Latents: 3 from the disk cache, 0 encoded" in caplog.text
    assert "Prompt embeddings: 1 from the disk cache, 0 encoded" in caplog.text
    assert all(torch.equal(first.latent_cache[k], second.latent_cache[k]) for k in first.latent_cache)
    assert all(torch.equal(first.text_cache[k], second.text_cache[k]) for k in first.text_cache)


def test_an_edited_caption_encodes_only_the_new_prompt(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 2)
    _cached(workspace)
    fakes.text.prompts.clear()
    (workspace.data / "p" / "img0.txt").write_text("a brand new caption", encoding="utf-8")

    trainer = _cached(workspace)

    assert fakes.text.prompts == ["a brand new caption"]
    assert fakes.encoder.encoded == 2 and "a brand new caption" in trainer.text_cache


def test_an_edited_image_encodes_only_that_image(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 3)
    _cached(workspace)
    write_image(workspace.data / "p" / "img1.png", shade=77)

    _cached(workspace)

    assert fakes.encoder.encoded == 4


def test_a_changed_resolution_encodes_the_latents_again_but_not_the_prompts(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 2)
    _cached(workspace)
    fakes.text.prompts.clear()

    _cached(workspace, resolution=48)

    assert (fakes.encoder.encoded, fakes.text.prompts) == (4, [])


def test_a_changed_vae_file_encodes_the_latents_again_but_not_the_prompts(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 2)
    _with_model_files(workspace)
    _cached(workspace)
    fakes.text.prompts.clear()
    (workspace.root / "vae.safetensors").write_bytes(b"vae-2")

    _cached(workspace)

    assert (fakes.encoder.encoded, fakes.text.prompts) == (4, [])


def test_a_changed_text_encoder_file_encodes_the_prompts_again_but_not_the_latents(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images("p", 2)
    _with_model_files(workspace)
    _cached(workspace)
    fakes.text.prompts.clear()
    (workspace.root / "text.safetensors").write_bytes(b"text-2")

    _cached(workspace)

    assert fakes.encoder.encoded == 2 and "a photo of zed" in fakes.text.prompts


def test_a_moved_dataset_keeps_its_cache(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 2)
    _cached(workspace)
    shutil.move(workspace.data, workspace.root / "moved")

    _cached(workspace, path=str(workspace.root / "moved"))

    assert (fakes.vae_loads, len(fakes.text_loads)) == (1, 1)


def test_with_the_cache_off_nothing_is_written_and_every_run_encodes(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images("p", 2)
    cached_lora(workspace.lora_config())
    cached_lora(workspace.lora_config())

    assert not (workspace.data / "p" / CACHE_DIR_NAME).exists()
    assert (fakes.vae_loads, fakes.encoder.encoded) == (2, 4)


def test_an_unreadable_entry_is_reported_and_encoded_again(
    workspace: Workspace, fakes: Fakes, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images("p", 2)
    _cached(workspace)
    broken = _entries(workspace.data / "p", "latents")[0]
    broken.write_bytes(b"truncated")

    with caplog.at_level(logging.WARNING):
        trainer = _cached(workspace)

    assert "Unreadable cache entry" in caplog.text and fakes.encoder.encoded == 3
    assert len(trainer.latent_cache) == 2 and broken.stat().st_size > len(b"truncated")


def test_a_caption_that_a_new_folder_shares_is_copied_into_its_cache_without_encoding(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images("a", 1, caption="shared words")
    _cached(workspace)
    workspace.add_images("b", 1, caption="shared words")
    loads = len(fakes.text_loads)

    _cached(workspace)

    assert len(fakes.text_loads) == loads
    assert len(_entries(workspace.data / "b", "text")) == 1
