"""Step 1 of a run: the dataset scan, the latent cache, the prompt cache and what happens to bad files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from maltrainer.data.flux3_dataset import load_flux3_image
from train_visage_lora import VisageLoRATrainer

from .fakes import Fakes
from .runs import run_lora
from .world import Workspace, write_image


def _cached(ws: Workspace, **sections: Any) -> VisageLoRATrainer:
    trainer = VisageLoRATrainer(ws.config(**sections))
    trainer.precache_dataset()
    return trainer


def test_every_image_gets_a_latent_and_every_distinct_caption_one_set_of_states(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images(2)
    write_image(workspace.data / "other.png", "a photo of zed smiling")

    trainer = _cached(workspace)

    assert [Path(key).name for key, _ in trainer.samples] == ["img0.png", "img1.png", "other.png"]
    assert {latent.shape for latent in trainer.latent_cache.values()} == {(8, 2, 2)}
    assert sorted(trainer.text_cache) == ["a photo of zed", "a photo of zed smiling"]
    assert sorted(fakes.text.prompts) == ["a photo of zed", "a photo of zed smiling"]
    assert fakes.vae_loads == 1 and fakes.text_loads == [(str(workspace.root / "text.safetensors"), None)]


def test_the_trainers_caption_is_the_sidecar_text_or_the_trigger_and_default(workspace: Workspace) -> None:
    write_image(workspace.data / "a.png", "zed on a bench")
    write_image(workspace.data / "b.png")

    trainer = _cached(workspace, dataset={"default_caption": "a portrait"})

    assert [caption for _, caption in trainer.samples] == ["zed on a bench", "zed a portrait"]


def test_the_tokenizer_path_reaches_the_text_encoder_loader(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images(1)

    _cached(workspace, model={"tokenizer_path": "tok"})

    assert fakes.text_loads[0][1] == "tok"


def test_images_are_centre_cropped_to_a_square_unless_the_bucket_mode_keeps_the_aspect(
    workspace: Workspace, fakes: Fakes
) -> None:
    write_image(workspace.data / "wide.png", "zed", size=(64, 32))

    square = _cached(workspace, dataset={"resolution": 48})
    bucket = _cached(workspace, dataset={"resolution": 48, "aspect_mode": "BUCKET"})

    assert fakes.vae.shapes == [(1, 3, 48, 48), (1, 3, 32, 64)]
    assert next(iter(square.latent_cache.values())).shape == (8, 3, 3)
    assert next(iter(bucket.latent_cache.values())).shape == (8, 2, 4)


def test_the_resolution_defaults_to_the_models_256(workspace: Workspace, fakes: Fakes) -> None:
    write_image(workspace.data / "big.png", "zed", size=(512, 384))
    config = workspace.config()
    del config["dataset"]["resolution"]

    VisageLoRATrainer(config).precache_dataset()

    assert fakes.vae.shapes == [(1, 3, 256, 256)]


def test_caption_dropout_needs_no_empty_prompt_because_dropped_states_are_zeros(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = _cached(workspace, train={"caption_dropout": 0.5})

    assert sorted(trainer.text_cache) == ["a photo of zed"]


def test_an_unreadable_image_is_skipped_with_a_warning_and_the_rest_train(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    workspace.add_images(2)
    (workspace.data / "broken.png").write_bytes(b"not an image")

    with caplog.at_level(logging.WARNING):
        trainer = run_lora(workspace.config())

    assert len(trainer.samples) == 2 and len(trainer.latent_cache) == 2
    assert "Skipping broken.png:" in caplog.text
    assert (workspace.out / "visage.safetensors").exists()


def test_a_folder_without_images_is_refused_before_any_model_loads(workspace: Workspace, fakes: Fakes) -> None:
    with pytest.raises(ValueError, match=r"No images found under .*data"):
        run_lora(workspace.config())

    assert fakes.vae_loads == 0


def test_a_folder_of_nothing_but_unreadable_images_is_refused(workspace: Workspace) -> None:
    (workspace.data).mkdir()
    (workspace.data / "broken.png").write_bytes(b"not an image")

    with pytest.raises(ValueError, match="No usable images"):
        run_lora(workspace.config())


def test_the_cache_logs_the_counts(workspace: Workspace, caplog: pytest.LogCaptureFixture) -> None:
    workspace.add_images(2)

    with caplog.at_level(logging.INFO):
        _cached(workspace)

    assert "Pre-cached 2 images and 1 prompt states." in caplog.text


def test_the_cached_latent_is_the_vaes_encoding_of_the_loaded_image_on_the_cpu(
    workspace: Workspace, fakes: Fakes
) -> None:
    workspace.add_images(1)

    trainer = _cached(workspace)

    key = trainer.samples[0][0]
    expected = fakes.vae.encode(load_flux3_image(Path(key), 32, "square").unsqueeze(0))[0]
    assert torch.equal(trainer.latent_cache[key], expected)
    assert trainer.latent_cache[key].device.type == "cpu"
