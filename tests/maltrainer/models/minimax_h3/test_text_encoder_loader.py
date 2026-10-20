from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from maltrainer.models.minimax_h3 import text_encoder as te
from tests.maltrainer.models.minimax_h3.h3_text import FakeTokenizer

CPU = torch.device("cpu")


class _VisionRotary(nn.Module):
    def __init__(self, dim: int = 0) -> None:
        super().__init__()
        self.dim = dim


class _TextRotary(nn.Module):
    def __init__(self, config: object = None, device: object = None) -> None:
        super().__init__()
        self.config, self.device_arg = config, device


class _FakeQwen(nn.Module):
    instances: ClassVar[list[_FakeQwen]] = []

    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.visual = nn.Module()
        self.visual.rotary_pos_emb = _VisionRotary()
        self.language_model = nn.Module()
        self.language_model.norm = nn.Module()
        self.language_model.rotary_emb = _TextRotary()
        self.loaded: dict[str, torch.Tensor] | None = None
        self.strict: bool | None = None
        _FakeQwen.instances.append(self)

    def load_state_dict(self, state, strict=True, **_options):
        self.loaded, self.strict = state, strict


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=64, use_cache=True, hidden_size=32),
        vision_config=SimpleNamespace(hidden_size=48, num_heads=4),
    )
    seen = SimpleNamespace(config=config, tokenizer=FakeTokenizer(), asset_dirs=[], quantized_calls=[], quantized=False)
    seen.tokenizer.pad_token = None
    seen.tokenizer.pad_token_id = None

    def load_config(path: str, **_options: object) -> SimpleNamespace:
        seen.asset_dirs.append(path)
        return config

    def load_tokenizer(path: str, **_options: object) -> FakeTokenizer:
        seen.asset_dirs.append(path)
        return seen.tokenizer

    monkeypatch.setattr(te, "Qwen3VLModel", _FakeQwen)
    monkeypatch.setattr(te, "Qwen3VLConfig", SimpleNamespace(from_pretrained=load_config))
    monkeypatch.setattr(te, "AutoTokenizer", SimpleNamespace(from_pretrained=load_tokenizer))
    monkeypatch.setattr(te, "has_comfy_quantized_layers", lambda _path: seen.quantized)
    monkeypatch.setattr(te, "load_comfy_quantized_state_dict", lambda *a, **k: seen.quantized_calls.append((a, k)))
    _FakeQwen.instances.clear()
    return seen


def _checkpoint(tmp_path: Path) -> Path:
    names = ("model.layers.0.q.weight", "visual.blocks.0.w", "model.norm.weight", "lm_head.weight")
    path = tmp_path / "te.safetensors"
    save_file({name: torch.zeros(1) for name in names}, str(path))
    return path


def test_the_loader_truncates_the_text_tower_to_layer_50_and_rebuilds_the_rotary_tables(
    fakes: SimpleNamespace, tmp_path: Path
) -> None:
    loaded = te.load_minimax_h3_text_encoder(_checkpoint(tmp_path), device="cpu")

    model = loaded.model
    assert fakes.config.text_config.num_hidden_layers == te.MINIMAX_H3_TEXT_LAYERS == 50
    assert fakes.config.text_config.use_cache is False
    assert isinstance(model.language_model.norm, nn.Identity)
    assert model.visual.rotary_pos_emb.dim == 48 // 4 // 2
    assert model.language_model.rotary_emb.config is fakes.config.text_config
    assert model.language_model.rotary_emb.device_arg == CPU
    assert isinstance(loaded, te.MiniMaxH3TextEncoder) and loaded.model is _FakeQwen.instances[0]
    assert loaded.tokenizer is fakes.tokenizer
    assert not loaded.training
    assert fakes.asset_dirs == [str(te.BUNDLED_TOKENIZER_ASSETS)] * 2


def test_an_explicit_tokenizer_directory_is_the_one_the_tokenizer_is_loaded_from(
    fakes: SimpleNamespace, tmp_path: Path
) -> None:
    custom = tmp_path / "tok"
    custom.mkdir()
    (custom / "tokenizer.json").write_text("{}")

    te.load_minimax_h3_text_encoder(_checkpoint(tmp_path), tokenizer_path=custom, device="cpu")

    assert fakes.asset_dirs == [str(custom)] * 2


def test_a_tokenizer_without_a_pad_token_pads_with_bos(fakes: SimpleNamespace, tmp_path: Path) -> None:
    te.load_minimax_h3_text_encoder(_checkpoint(tmp_path), device="cpu")

    assert fakes.tokenizer.pad_token_id == te.QWEN3VL_BOS_TOKEN_ID


def test_an_existing_pad_token_is_kept(fakes: SimpleNamespace, tmp_path: Path) -> None:
    fakes.tokenizer.pad_token, fakes.tokenizer.pad_token_id = "<pad>", 5

    te.load_minimax_h3_text_encoder(_checkpoint(tmp_path), device="cpu")

    assert fakes.tokenizer.pad_token_id == 5


def test_a_plain_checkpoint_is_remapped_onto_the_towers_and_other_tensors_dropped(
    fakes: SimpleNamespace, tmp_path: Path
) -> None:
    loaded = te.load_minimax_h3_text_encoder(_checkpoint(tmp_path), device="cpu")

    assert sorted(loaded.model.loaded) == [
        "language_model.layers.0.q.weight",
        "language_model.norm.weight",
        "visual.blocks.0.w",
    ]
    assert loaded.model.strict is False
    assert fakes.quantized_calls == []


def test_a_comfy_quantized_checkpoint_goes_through_the_quantized_loader(fakes: SimpleNamespace, tmp_path: Path) -> None:
    fakes.quantized = True
    path = tmp_path / "nvfp4.safetensors"

    loaded = te.load_minimax_h3_text_encoder(path, device="cpu", dtype=torch.float16)

    ((args, kwargs),) = fakes.quantized_calls
    assert args == (loaded.model, path)
    assert kwargs == {"key_map": te._text_encoder_key, "output_dtype": torch.float16}
    assert loaded.model.loaded is None


@pytest.mark.usefixtures("fakes")
def test_a_missing_tokenizer_directory_stops_the_load_before_anything_is_built(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"tokenizer\.json"):
        te.load_minimax_h3_text_encoder(_checkpoint(tmp_path), tokenizer_path=tmp_path / "nope", device="cpu")

    assert _FakeQwen.instances == []
