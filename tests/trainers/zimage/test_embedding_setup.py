"""Model loading, the embedding's creation and the dataset of the textual-inversion trainer."""

from __future__ import annotations

import pytest
import torch

import train_zimage_embedding as embedding

from .fakes import Fakes
from .runs import prepared_embedding, run_embedding
from .tiny import WORDS, build_text_encoder, embedding_table
from .world import Workspace, write_image


def _loaded(ws: Workspace, **sections: object) -> embedding.ZImageEmbeddingTrainer:
    trainer = embedding.ZImageEmbeddingTrainer(ws.embedding_config(**sections))
    trainer.load_model()
    return trainer


def test_every_part_comes_from_the_one_model_path(workspace: Workspace, fakes: Fakes) -> None:
    _loaded(workspace, model={"name_or_path": "some/model"})

    assert [(x.kind, x.path, x.subfolder, x.dtype) for x in fakes.loads] == [
        ("transformer", "some/model", "transformer", torch.float32),
        ("text_encoder", "some/model", "text_encoder", torch.float32),
        ("tokenizer", "some/model", "tokenizer", None),
        ("vae", "some/model", "vae", torch.float32),
    ]


def test_only_the_input_embedding_table_may_learn_and_the_transformer_checkpoints_its_gradients(
    workspace: Workspace,
) -> None:
    trainer = _loaded(workspace)

    trainable = {n for n, p in trainer.text_encoder.named_parameters() if p.requires_grad}
    assert trainable == {"model.embed_tokens.weight"}
    assert not any(p.requires_grad for m in (trainer.transformer, trainer.vae) for p in m.parameters())
    assert trainer.transformer.gradient_checkpointing is True


def test_the_vae_stays_float32_while_the_transformer_and_text_encoder_take_the_configured_dtype(
    workspace: Workspace,
) -> None:
    trainer = _loaded(workspace, dtype="bfloat16")

    assert next(trainer.transformer.parameters()).dtype == torch.bfloat16
    assert trainer.text_encoder.get_input_embeddings().weight.dtype == torch.bfloat16
    assert next(trainer.vae.parameters()).dtype == torch.float32


def test_a_bfloat16_run_trains_the_trigger_row_in_bfloat16(workspace: Workspace) -> None:
    workspace.add_images(2)

    trainer = run_embedding(workspace.embedding_config(dtype="bfloat16"))

    table = trainer.text_encoder.get_input_embeddings().weight
    assert table.dtype == torch.bfloat16
    assert not torch.equal(table[len(WORDS)], embedding_table(build_text_encoder()).to(torch.bfloat16)[5])


def test_the_trigger_and_its_extra_vectors_become_new_tokens_and_the_table_grows_to_hold_them(
    workspace: Workspace,
) -> None:
    workspace.add_images(1)

    trainer = prepared_embedding(workspace.embedding_config(embedding={"trigger": "tok", "num_vectors": 3}))

    assert trainer.embedding.token_ids == [len(WORDS), len(WORDS) + 1, len(WORDS) + 2]
    assert trainer.tokenizer.convert_tokens_to_ids(["tok", "tok_1", "tok_2"]) == trainer.embedding.token_ids
    assert trainer.text_encoder.get_input_embeddings().weight.shape[0] == len(trainer.tokenizer) == len(WORDS) + 3


def test_the_original_table_is_snapshotted_after_the_new_rows_were_initialised(workspace: Workspace) -> None:
    workspace.add_images(1)

    trainer = prepared_embedding(workspace.embedding_config())

    table = trainer.text_encoder.get_input_embeddings().weight
    assert torch.equal(trainer.original_embeddings, table.detach())
    assert trainer.original_embeddings.data_ptr() != table.data_ptr()
    assert torch.equal(trainer.original_embeddings[len(WORDS)], trainer.original_embeddings[WORDS.index("a")])


def test_without_an_embedding_section_the_trigger_is_sks_with_one_vector_from_a_photo(workspace: Workspace) -> None:
    workspace.add_images(1)
    config = workspace.embedding_config()
    del config["embedding"]

    trainer = prepared_embedding(config)

    assert (trainer.embedding.trigger_token, trainer.embedding.num_vectors, trainer.embedding.init_text) == (
        "sks",
        1,
        "a photo",
    )
    assert trainer.train_dataset.captions == ["sks a photo of person"]


@pytest.mark.parametrize(("known", "vectors", "added"), [(["sks"], 1, 0), (["sks"], 2, 1), (["sks_1"], 2, 1)])
def test_a_trigger_the_tokenizer_already_has_is_refused(
    workspace: Workspace, known: list[str], vectors: int, added: int
) -> None:
    trainer = _loaded(workspace, embedding={"num_vectors": vectors})
    trainer.tokenizer.add_tokens(known)

    with pytest.raises(ValueError, match=rf"Expected to add {vectors} tokens, but added {added}\. Token 'sks' may"):
        trainer.create_embedding()


def test_the_dataset_prefixes_the_embedding_trigger_and_ignores_the_default_caption(workspace: Workspace) -> None:
    write_image(workspace.data / "empty.png")
    write_image(workspace.data / "named.png", "a photo of person")

    trainer = prepared_embedding(workspace.embedding_config(dataset={"repeats": 2}))

    captions = dict(
        zip((p.stem for p in trainer.train_dataset.image_paths), trainer.train_dataset.captions, strict=True)
    )
    assert captions == {"empty": "sks", "named": "sks a photo of person"}
    assert len(trainer.train_dataset) == 4


def test_the_dataloader_batches_the_samples_and_a_square_crop_is_the_default(workspace: Workspace) -> None:
    workspace.add_images(3)

    trainer = prepared_embedding(workspace.embedding_config(train={"batch_size": 2}))

    assert len(trainer.train_dataloader) == 2
    assert next(iter(trainer.train_dataloader))["pixel_values"].shape == (2, 3, 16, 16)
