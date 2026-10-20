"""The caption embeddings: which prompts are encoded, in what order, and how."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from .fakes import Fakes
from .media import write_image
from .runs import cached
from .world import Workspace


def _two_captions(ws: Workspace) -> None:
    write_image(ws.data / "a.png", "a red car")
    write_image(ws.data / "b.png", "a red car")
    write_image(ws.data / "c.png", "two dogs")


def test_every_unique_caption_is_encoded_once_in_sorted_order(workspace: Workspace, fakes: Fakes) -> None:
    _two_captions(workspace)

    trainer = cached(workspace.config())

    assert fakes.text.prompts == ["a red car", "two dogs"]
    assert sorted(trainer.text_cache) == ["a red car", "two dogs"]


def test_embeddings_are_kept_on_the_cpu_as_one_row_per_prompt(workspace: Workspace) -> None:
    _two_captions(workspace)

    trainer = cached(workspace.config())

    assert trainer.text_cache["a red car"].shape == (1, 3, 6)
    assert trainer.text_cache["two dogs"].shape == (1, 2, 6)
    assert {t.device.type for t in trainer.text_cache.values()} == {"cpu"}


def test_the_embeddings_follow_the_run_dtype(workspace: Workspace, fakes: Fakes) -> None:
    _two_captions(workspace)

    trainer = cached(workspace.config(dtype="bfloat16"))

    assert {t.dtype for t in trainer.text_cache.values()} == {torch.bfloat16}
    assert fakes.text_options[0][2] is torch.bfloat16


def test_the_tokenizer_path_goes_to_the_text_encoder_loader(workspace: Workspace, fakes: Fakes) -> None:
    _two_captions(workspace)

    cached(workspace.config(model={"tokenizer_path": "somewhere/tokenizer"}))

    assert fakes.text_options[0][0] == "somewhere/tokenizer"


@pytest.mark.parametrize("flag", ["do_guidance_loss", "do_differential_guidance"])
def test_guidance_loss_also_encodes_the_unconditional_prompt_as_one_bos_token(workspace: Workspace, flag: str) -> None:
    _two_captions(workspace)

    trainer = cached(workspace.config(train={flag: True}))

    assert trainer.text_cache[""].shape[1] == 1


def test_the_unconditional_prompt_can_be_set(workspace: Workspace) -> None:
    _two_captions(workspace)

    trainer = cached(workspace.config(train={"do_guidance_loss": True, "unconditional_prompt": "blurry bad"}))

    assert "blurry bad" in trainer.text_cache and "" not in trainer.text_cache


def test_the_new_flag_wins_over_the_old_alias(workspace: Workspace) -> None:
    _two_captions(workspace)

    trainer = cached(workspace.config(train={"do_guidance_loss": False, "do_differential_guidance": True}))

    assert "" not in trainer.text_cache


@pytest.mark.parametrize(("dropout", "cached_empty"), [(0.0, False), (None, False), (0.05, True)])
def test_caption_dropout_encodes_the_empty_prompt_so_a_dropped_caption_has_an_embedding(
    workspace: Workspace, dropout: Any, cached_empty: bool
) -> None:
    _two_captions(workspace)

    trainer = cached(workspace.config(train={"caption_dropout": dropout}))

    assert ("" in trainer.text_cache) is cached_empty
