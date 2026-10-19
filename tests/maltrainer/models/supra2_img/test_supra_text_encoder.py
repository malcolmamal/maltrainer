from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import T5Config, T5EncoderModel

from maltrainer.models.supra2_img.text_encoder import FlanT5Encoder, load_flan_t5

CPU = torch.device("cpu")
VOCAB = {"<pad>": 0, "</s>": 1, "<unk>": 2, "hello": 3, "world": 4}


@pytest.fixture
def assets(tmp_path: Path) -> dict[str, Path]:
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path, config_path = tmp_path / "tokenizer.json", tmp_path / "config.json"
    tokenizer.save(str(tokenizer_path))
    T5Config(vocab_size=200, d_model=16, d_kv=4, d_ff=32, num_layers=1, num_heads=4).to_json_file(str(config_path))
    return {"tokenizer": tokenizer_path, "config": config_path, "weights": tmp_path / "t5.safetensors"}


def _state(config_path: Path) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    model = T5EncoderModel(T5Config.from_json_file(str(config_path)))
    return {key: value.clone() for key, value in model.state_dict().items()}


def _load(assets: dict[str, Path], state: dict[str, torch.Tensor], ctx_len: int = 6) -> FlanT5Encoder:
    save_file(state, str(assets["weights"]))
    return load_flan_t5(assets["weights"], assets["config"], assets["tokenizer"], CPU, ctx_len=ctx_len)


def test_prompts_are_padded_to_the_context_length_and_encoded_to_float_states_and_a_mask(
    assets: dict[str, Path],
) -> None:
    encoder = _load(assets, _state(assets["config"]))

    hidden, mask = encoder.encode(["hello world", "hello"], CPU)

    assert hidden.shape == (2, 6, 16) and hidden.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert mask.tolist() == [[1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]]


def test_prompts_longer_than_the_context_are_truncated(assets: dict[str, Path]) -> None:
    encoder = _load(assets, _state(assets["config"]), ctx_len=3)

    hidden, mask = encoder.encode(["hello world hello world hello"], CPU)

    assert hidden.shape == (1, 3, 16) and mask.tolist() == [[1, 1, 1]]


def test_the_loaded_encoder_is_frozen_evaluates_and_carries_the_saved_weights(assets: dict[str, Path]) -> None:
    state = _state(assets["config"])

    encoder = _load(assets, state)

    assert not encoder.model.training and not any(p.requires_grad for p in encoder.model.parameters())
    assert torch.equal(
        encoder.model.encoder.block[0].layer[0].SelfAttention.q.weight,
        state["encoder.block.0.layer.0.SelfAttention.q.weight"],
    )
    assert encoder.ctx_len == 6


def test_decoder_and_lm_head_tensors_and_a_missing_token_table_are_tolerated(assets: dict[str, Path]) -> None:
    state = _state(assets["config"])
    del state["encoder.embed_tokens.weight"]
    state.update({"decoder.block.0.weight": torch.zeros(1), "lm_head.weight": torch.zeros(1)})

    encoder = _load(assets, state)

    assert encoder.encode(["hello"], CPU)[0].shape == (1, 6, 16)


@pytest.mark.parametrize(
    "damage",
    [lambda s: s.pop("encoder.final_layer_norm.weight"), lambda s: s.update({"rogue.weight": torch.zeros(1)})],
)
def test_other_missing_or_unexpected_tensors_are_an_error(assets: dict[str, Path], damage) -> None:
    state = _state(assets["config"])
    damage(state)

    with pytest.raises(RuntimeError, match="Flan-T5 load mismatch"):
        _load(assets, state)
