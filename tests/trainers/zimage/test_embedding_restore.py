"""Regularisation: after every step the table's rows other than the trigger's are put back."""

from __future__ import annotations

import torch

from .runs import prepared_embedding
from .tiny import WORDS
from .world import Workspace


def test_restoring_puts_back_every_row_except_the_triggers(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared_embedding(workspace.embedding_config(embedding={"num_vectors": 2}))
    table = trainer.text_encoder.get_input_embeddings().weight
    before = table.detach().clone()

    with torch.no_grad():
        table += 1.0
    trainer.restore_original_embeddings()

    assert torch.equal(table[: len(WORDS)], before[: len(WORDS)])
    assert torch.equal(table[len(WORDS) :], before[len(WORDS) :] + 1.0)


def test_restoring_leaves_a_table_that_was_not_touched_as_it_is(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared_embedding(workspace.embedding_config())
    before = trainer.text_encoder.get_input_embeddings().weight.detach().clone()

    trainer.restore_original_embeddings()

    assert torch.equal(trainer.text_encoder.get_input_embeddings().weight, before)


def test_the_snapshot_not_the_current_table_is_what_comes_back(workspace: Workspace) -> None:
    workspace.add_images(1)
    trainer = prepared_embedding(workspace.embedding_config())
    snapshot = trainer.original_embeddings
    trainer.original_embeddings = snapshot + 5.0

    trainer.restore_original_embeddings()

    table = trainer.text_encoder.get_input_embeddings().weight
    assert torch.equal(table[: len(WORDS)], snapshot[: len(WORDS)] + 5.0)
    assert torch.equal(table[len(WORDS)], snapshot[len(WORDS)])
