"""A textual-inversion run on the tiny model: only the trigger's embedding rows learn, the file holds them."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import train_zimage_embedding as embedding

from .fakes import Fakes
from .runs import run_embedding
from .tiny import WORDS, build_text_encoder, build_transformer, build_vae, embedding_table
from .world import Workspace, changed, named_weights

TRIGGER_ID = len(WORDS)
A_ID, PHOTO_ID = WORDS.index("a"), WORDS.index("photo")


def _trained(ws: Workspace, images: int = 3, **sections: Any) -> embedding.ZImageEmbeddingTrainer:
    ws.add_images(images)
    return run_embedding(ws.embedding_config(**sections))


def test_only_the_trigger_row_of_the_embedding_table_changes_and_every_other_weight_stays_identical(
    workspace: Workspace, fakes: Fakes
) -> None:
    trainer = _trained(workspace)

    table = trainer.text_encoder.get_input_embeddings().weight
    original = embedding_table(build_text_encoder())
    assert table.shape == (len(WORDS) + 1, 16)
    assert torch.equal(table[: len(WORDS)], original)
    assert not torch.equal(table[TRIGGER_ID], original[A_ID])
    others = {
        n: w
        for n, w in named_weights(trainer.text_encoder).items()
        if n not in ("model.embed_tokens.weight", "lm_head.weight")
    }
    reference = named_weights(build_text_encoder())
    assert not changed({n: reference[n] for n in others}, others)
    assert not changed(named_weights(build_transformer()), named_weights(trainer.transformer))
    assert not changed(named_weights(build_vae()), named_weights(fakes.parts["vae"]))


def test_the_saved_file_holds_the_learned_vectors_and_records_the_embedding_settings(workspace: Workspace) -> None:
    trainer = _trained(workspace, embedding={"trigger": "tok", "num_vectors": 3})

    path = workspace.out / "zimage_embedding_step_final.safetensors"
    saved = load_file(str(path))["embedding"]
    assert saved.shape == (3, 16) and torch.equal(saved, trainer.embedding.get_embeddings())
    with safe_open(str(path), framework="pt") as handle:
        assert handle.metadata() == {"step": "final", "trigger": "tok", "num_vectors": "3", "init_text": "a photo"}


def test_the_vectors_start_from_the_init_text_tokens_repeating_the_last_one(workspace: Workspace) -> None:
    workspace.add_images(1)
    config = workspace.embedding_config(embedding={"num_vectors": 3}, train={"learning_rate": 0.0})

    trainer = run_embedding(config)

    table = embedding_table(build_text_encoder())
    assert torch.equal(trainer.embedding.get_embeddings(), table[[A_ID, PHOTO_ID, PHOTO_ID]])


def test_a_checkpoint_is_written_every_save_every_steps_and_one_at_the_end(workspace: Workspace) -> None:
    trainer = _trained(workspace, train={"num_epochs": 2, "save_every": 4})

    names = sorted(p.name for p in workspace.out.glob("*.safetensors"))
    assert names == [f"zimage_embedding_step_{step}.safetensors" for step in (4, "final")]
    assert trainer.last_saved_path == str(workspace.out / "zimage_embedding_step_final.safetensors")


def test_the_text_encoder_sees_the_trigger_in_front_of_each_caption(workspace: Workspace, fakes: Fakes) -> None:
    workspace.add_images(1)
    (workspace.data / "img0.txt").unlink()

    run_embedding(workspace.embedding_config())

    assert fakes.prompts() == ["<|user|> sks <|assistant|> think"]


def test_the_transformer_is_fed_latents_the_flipped_timestep_and_text_hidden_states(
    workspace: Workspace, fakes: Fakes
) -> None:
    _trained(workspace, images=1)

    call = fakes.calls[0]
    assert call.x.shape == (1, 4, 1, 8, 8) and call.x.dtype == torch.float32
    assert call.t.shape == (1,) and 0.0 <= call.t.item() <= 1.0
    assert call.cap_feats.shape[0] == 1 and call.cap_feats.shape[2] == 16 and call.cap_mask is None


def test_a_transformer_with_a_cap_mask_parameter_gets_the_attention_mask_and_any_output_shape_works(
    workspace: Workspace, fakes: Fakes
) -> None:
    fakes.masked = True
    for output in ("tuple", "tensor", "plain"):
        fakes.output = output
        fakes.calls.clear()
        _trained(workspace, images=1)

        mask = fakes.calls[0].cap_mask
        assert mask is not None and mask.dtype == torch.bool and mask.all()


def test_a_batch_of_two_with_a_transposed_output_trains_too(workspace: Workspace, fakes: Fakes) -> None:
    fakes.output = "swapped"

    trainer = _trained(workspace, images=4, train={"batch_size": 2})

    assert fakes.calls[0].x.shape[0] == 2
    assert torch.isfinite(trainer.embedding.get_embeddings()).all()


def test_the_training_banner_names_the_device_dtype_and_model_sizes(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    _trained(workspace, images=1)

    assert capsys.readouterr().out.splitlines()[:5] == [
        "--- Training Configuration ---",
        "Device: CPU",
        "Dtype: torch.float32",
        "Transformer: 0.00B params",
        "Text Encoder: 0.00B params",
    ]


def test_the_same_seed_gives_the_same_vectors_and_another_seed_does_not(workspace: Workspace) -> None:
    def vectors(seed: int) -> torch.Tensor:
        return _trained(workspace, seed=seed).embedding.get_embeddings().detach().clone()

    first, again, other = vectors(7), vectors(7), vectors(8)

    assert torch.equal(first, again)
    assert not torch.equal(first, other)
