"""The loader's own logic, with the Qwen3-VL model and tokenizer faked.

Building the real model is out of reach on CPU in a unit test (and ``load_state_dict`` into the
``init_empty_weights`` model only warns, see the report), so the fakes record what the loader asks of them.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import maltrainer.models.minimax_h3.text_encoder as te
from maltrainer.models.minimax_h3.text_encoder import (
    MINIMAX_H3_TEXT_LAYERS,
    QWEN3VL_BOS_TOKEN_ID,
    MiniMaxH3TextEncoder,
    load_minimax_h3_text_encoder,
)

CPU = torch.device("cpu")


class Rotary(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.args, self.kwargs = args, kwargs


class FakeVLModel(nn.Module):
    built: ClassVar[list] = []

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.visual = nn.Module()
        self.visual.rotary_pos_emb = Rotary(1)
        self.language_model = nn.Module()
        self.language_model.norm = nn.LayerNorm(2)
        self.language_model.rotary_emb = Rotary(config.text_config)
        self.loaded: dict | None = None
        FakeVLModel.built.append(self)

    def load_state_dict(self, state_dict, *_args, **_kwargs):
        self.loaded = dict(state_dict)
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


class FakeTokenizer:
    def __init__(self, pad_token: str | None) -> None:
        self.pad_token = pad_token
        self.pad_token_id = 7 if pad_token else None


class FakeAutoTokenizer:
    pad_token: str | None = None
    loaded_from: ClassVar[list] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> FakeTokenizer:
        cls.loaded_from.append((path, kwargs))
        return FakeTokenizer(cls.pad_token)


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeVLModel.built = []
    FakeAutoTokenizer.pad_token, FakeAutoTokenizer.loaded_from = None, []
    monkeypatch.setattr(te, "Qwen3VLModel", FakeVLModel)
    monkeypatch.setattr(te, "AutoTokenizer", FakeAutoTokenizer)


def _plain_checkpoint(tmp_path: Path) -> Path:
    tensors = {
        "model.layers.0.mlp.up_proj.weight": torch.ones(2),
        "model.embed_tokens.weight": torch.ones(3),
        "model.norm.weight": torch.ones(1),
        "visual.blocks.0.attn.qkv.weight": torch.ones(4),
        "lm_head.weight": torch.ones(5),
    }
    path = tmp_path / "te.safetensors"
    save_file(tensors, str(path))
    return path


def test_the_loader_truncates_the_text_tower_to_layer_fifty_and_disables_the_kv_cache(tmp_path: Path) -> None:
    load_minimax_h3_text_encoder(_plain_checkpoint(tmp_path), device="cpu")

    config = FakeVLModel.built[0].config
    assert MINIMAX_H3_TEXT_LAYERS == 50
    assert config.text_config.num_hidden_layers == 50 and config.text_config.use_cache is False


def test_the_final_text_norm_is_removed_so_the_unnormalised_hidden_state_is_returned(tmp_path: Path) -> None:
    load_minimax_h3_text_encoder(_plain_checkpoint(tmp_path), device="cpu")

    assert isinstance(FakeVLModel.built[0].language_model.norm, nn.Identity)


def test_a_plain_checkpoint_is_mapped_onto_the_vision_and_language_towers_and_other_keys_are_ignored(
    tmp_path: Path,
) -> None:
    load_minimax_h3_text_encoder(_plain_checkpoint(tmp_path), device="cpu")

    assert set(FakeVLModel.built[0].loaded) == {
        "language_model.layers.0.mlp.up_proj.weight",
        "language_model.embed_tokens.weight",
        "language_model.norm.weight",
        "visual.blocks.0.attn.qkv.weight",
    }


def test_rotary_embeddings_are_rebuilt_on_the_target_device_from_the_vision_and_text_configs(tmp_path: Path) -> None:
    load_minimax_h3_text_encoder(_plain_checkpoint(tmp_path), device="cpu")

    model = FakeVLModel.built[0]
    vision = model.config.vision_config
    assert model.visual.rotary_pos_emb.args == (vision.hidden_size // vision.num_heads // 2,)
    assert model.language_model.rotary_emb.args == (model.config.text_config,)
    assert model.language_model.rotary_emb.kwargs == {"device": CPU}


def test_the_loaded_wrapper_is_frozen_eval_and_carries_the_tokenizer(tmp_path: Path) -> None:
    encoder = load_minimax_h3_text_encoder(_plain_checkpoint(tmp_path), device="cpu")

    assert isinstance(encoder, MiniMaxH3TextEncoder)
    assert encoder.model is FakeVLModel.built[0] and not encoder.model.training
    assert isinstance(encoder.tokenizer, FakeTokenizer)


def test_a_tokenizer_without_a_pad_token_pads_with_the_qwen_bos_id(tmp_path: Path) -> None:
    encoder = load_minimax_h3_text_encoder(_plain_checkpoint(tmp_path), device="cpu")

    assert encoder.tokenizer.pad_token_id == QWEN3VL_BOS_TOKEN_ID


def test_a_tokenizer_that_already_pads_keeps_its_pad_id(tmp_path: Path) -> None:
    FakeAutoTokenizer.pad_token = "<pad>"

    encoder = load_minimax_h3_text_encoder(_plain_checkpoint(tmp_path), device="cpu")

    assert encoder.tokenizer.pad_token_id == 7


def test_the_tokenizer_comes_from_the_bundled_assets_unless_a_directory_is_given(tmp_path: Path) -> None:
    load_minimax_h3_text_encoder(_plain_checkpoint(tmp_path), device="cpu")
    custom = tmp_path / "tok"
    custom.mkdir()
    (custom / "tokenizer.json").write_text("{}")
    load_minimax_h3_text_encoder(_plain_checkpoint(tmp_path), tokenizer_path=custom, device="cpu")

    assert [p for p, _ in FakeAutoTokenizer.loaded_from] == [str(te.BUNDLED_TOKENIZER_ASSETS), str(custom)]


def test_a_missing_tokenizer_json_stops_the_load_before_any_model_is_built(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("No tokenizer.json")):
        load_minimax_h3_text_encoder(_plain_checkpoint(tmp_path), tokenizer_path=tmp_path / "nowhere", device="cpu")

    assert FakeVLModel.built == []


def test_a_comfy_quantized_checkpoint_is_loaded_through_the_quantized_loader_with_the_key_map_and_dtype(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(te, "has_comfy_quantized_layers", lambda _path: True)
    monkeypatch.setattr(te, "load_comfy_quantized_state_dict", lambda *a, **kw: calls.append((a, kw)))
    path = tmp_path / "te_nvfp4.safetensors"

    load_minimax_h3_text_encoder(path, device="cpu", dtype=torch.float16)

    ((args, kwargs),) = calls
    assert args == (FakeVLModel.built[0], path)
    assert kwargs == {"key_map": te._text_encoder_key, "output_dtype": torch.float16}
    assert FakeVLModel.built[0].loaded is None  # the plain path was not taken
