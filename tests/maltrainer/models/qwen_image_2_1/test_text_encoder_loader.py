from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maltrainer.models.qwen_image_2_1 import text_encoder as te
from maltrainer.models.qwen_image_2_1.text_encoder import Int8Embedding, Qwen3VLTextEncoder

from ._tiny import int8_checkpoint

CPU = torch.device("cpu")
TOKENIZER = object()


class _Tokenizers:
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, path: str, **options: object) -> object:
        cls.calls.append((path, options))
        return TOKENIZER


class _Config:
    created: SimpleNamespace

    @classmethod
    def from_pretrained(cls, path: str) -> SimpleNamespace:
        text = SimpleNamespace(vocab_size=8)
        vision = SimpleNamespace(hidden_size=8, num_heads=2)
        cls.created = SimpleNamespace(text_config=text, vision_config=vision, path=path)
        return cls.created


class _VisionRotary(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim


class _TextRotary(nn.Module):
    def __init__(self, config: SimpleNamespace, device: torch.device | None = None) -> None:
        super().__init__()
        self.config, self.device = config, device


class _Model(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.visual = nn.Module()
        self.visual.proj = nn.Linear(4, 4, bias=False)
        self.visual.rotary_pos_emb = _VisionRotary(1)
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(8, 4)
        self.language_model.norm = nn.LayerNorm(4)
        self.language_model.proj = nn.Linear(8, 4, bias=False)
        self.language_model.rotary_emb = _TextRotary(config)


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    _Tokenizers.calls = []
    monkeypatch.setattr(te, "AutoTokenizer", _Tokenizers)
    monkeypatch.setattr(te, "Qwen3VLConfig", _Config)
    monkeypatch.setattr(te, "Qwen3VLModel", _Model)


def _checkpoint(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(0)
    state = {
        "model.embed_tokens.weight": torch.zeros(8, 4096),
        "model.proj.weight": torch.zeros(4, 8),
        "model.visual.proj.weight": torch.randn(4, 4, generator=generator),
        "lm_head.weight": torch.zeros(8, 4),
    }
    quantized = {"model.embed_tokens": 4, "model.proj": 4}
    path = tmp_path / "te.safetensors"
    return path, int8_checkpoint(path, state, quantized)


def test_a_missing_checkpoint_is_reported_before_anything_is_loaded(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Text encoder checkpoint not found"):
        te.load_qwen_image_2_1_text_encoder(tmp_path / "absent.safetensors", device=CPU)

    assert _Tokenizers.calls == []


def test_the_bundled_assets_are_the_default_and_a_given_asset_dir_wins(tmp_path: Path) -> None:
    path, _ = _checkpoint(tmp_path)

    te.load_qwen_image_2_1_text_encoder(str(path), device=CPU)
    te.load_qwen_image_2_1_text_encoder(path, asset_dir=tmp_path / "assets", device=CPU)

    default, custom = (Path(call[0]) for call in _Tokenizers.calls)
    assert default.parts[-3:] == ("minimax_h3", "assets", "text_encoder")
    assert default.is_dir()
    assert custom == tmp_path / "assets"
    assert _Tokenizers.calls[0][1] == {"trust_remote_code": True}


def test_the_config_is_forced_to_the_8b_text_encoder_shape(tmp_path: Path) -> None:
    te.load_qwen_image_2_1_text_encoder(_checkpoint(tmp_path)[0], device=CPU)

    text = _Config.created.text_config
    assert (text.num_hidden_layers, text.hidden_size, text.intermediate_size) == (36, 4096, 12288)
    assert (text.num_attention_heads, text.num_key_value_heads, text.use_cache) == (32, 8, False)
    assert _Config.created.vision_config.out_hidden_size == 4096


def test_the_embedding_table_is_replaced_by_an_int8_embedding_holding_the_file_weights(tmp_path: Path) -> None:
    path, stored = _checkpoint(tmp_path)

    encoder = te.load_qwen_image_2_1_text_encoder(path, device=CPU)

    table = encoder.model.language_model.embed_tokens
    assert isinstance(table, Int8Embedding)
    assert torch.equal(table.weight, stored["model.embed_tokens"])
    assert table.int8_convrot_groupsize.item() == 4
    assert table.scale_weight.shape == (8, 1)


def test_the_loader_returns_a_frozen_eval_wrapper_around_the_tokenizer_and_model(tmp_path: Path) -> None:
    encoder = te.load_qwen_image_2_1_text_encoder(_checkpoint(tmp_path)[0], device=CPU)

    assert isinstance(encoder, Qwen3VLTextEncoder)
    assert encoder.tokenizer is TOKENIZER
    assert not encoder.model.training
    assert not any(p.requires_grad for p in encoder.model.parameters())
    assert isinstance(encoder.model.language_model.norm, nn.Identity)


def test_checkpoint_keys_land_under_language_model_and_visual_and_the_lm_head_is_dropped(tmp_path: Path) -> None:
    path, stored = _checkpoint(tmp_path)
    encoder = te.load_qwen_image_2_1_text_encoder(path, device=CPU)

    language = encoder.model.language_model
    assert torch.equal(language.proj.weight, stored["model.proj"])
    assert encoder.model.visual.proj.weight.dtype == torch.float32
    assert language.proj.int8_convrot_groupsize.item() == 4


@pytest.mark.parametrize(("requested", "expected"), [("bf16", "bf16"), ("fp32", "fp32")])
def test_the_gradient_precision_reaches_the_quantized_linears(tmp_path: Path, requested: str, expected: str) -> None:
    encoder = te.load_qwen_image_2_1_text_encoder(_checkpoint(tmp_path)[0], device=CPU, grad_precision=requested)

    assert encoder.model.language_model.proj.int8_convrot_grad_precision == expected


def test_the_rotary_modules_are_rebuilt_from_the_config_on_the_target_device(tmp_path: Path) -> None:
    encoder = te.load_qwen_image_2_1_text_encoder(_checkpoint(tmp_path)[0], device=CPU)

    assert encoder.model.visual.rotary_pos_emb.dim == (8 // 2) // 2
    assert encoder.model.language_model.rotary_emb.config is _Config.created.text_config
    assert encoder.model.language_model.rotary_emb.device == CPU


def test_a_checkpoint_without_the_embedding_table_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "te.safetensors"
    int8_checkpoint(path, {"model.proj.weight": torch.zeros(4, 8)}, {"model.proj": 4})

    with pytest.raises(KeyError, match="embed_tokens"):
        te.load_qwen_image_2_1_text_encoder(path, device=CPU)
