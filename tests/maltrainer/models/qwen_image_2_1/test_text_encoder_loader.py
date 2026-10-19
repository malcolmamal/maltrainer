from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from maltrainer.models.qwen_image_2_1 import text_encoder as te

from ._tiny import int8_checkpoint

CPU = torch.device("cpu")
VOCAB, GROUP = 6, 4
RECEIVED: dict = {}


class FakeRotary(nn.Module):
    def __init__(self, arg: object, device: object = None) -> None:
        super().__init__()
        self.arg, self.device_arg = arg, device


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, 4)
        self.layers = nn.ModuleList([nn.ModuleDict({"proj": nn.Linear(8, 4, bias=False)})])
        self.norm = nn.LayerNorm(4)
        self.rotary_emb = FakeRotary(None)


class FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rotary_pos_emb = FakeRotary(0)


class FakeQwen3VLModel(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        RECEIVED["config"] = config
        self.language_model = FakeLanguageModel()
        self.visual = FakeVisual()


def _config() -> SimpleNamespace:
    text = SimpleNamespace(vocab_size=VOCAB, hidden_size=1, num_hidden_layers=1)
    vision = SimpleNamespace(hidden_size=64, num_heads=16, out_hidden_size=1)
    return SimpleNamespace(text_config=text, vision_config=vision)


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    RECEIVED.clear()
    monkeypatch.setattr(te, "Qwen3VLModel", FakeQwen3VLModel)
    monkeypatch.setattr(te.Qwen3VLConfig, "from_pretrained", lambda _path: _config())

    def tokenizer(path: str, **_: object) -> str:
        RECEIVED["tokenizer_path"] = path
        return "tokenizer"

    monkeypatch.setattr(te.AutoTokenizer, "from_pretrained", tokenizer)


def _checkpoint(path: Path) -> dict[str, torch.Tensor]:
    """Comfy INT8 ConvRot file: a quantised linear, a quantised embedding (hidden is forced to 4096), an lm_head."""
    state = {
        "model.layers.0.proj.weight": torch.zeros(4, 8),
        "model.embed_tokens.weight": torch.zeros(VOCAB, 4096),
        "lm_head.weight": torch.zeros(2, 2),
    }
    stored = int8_checkpoint(path, state, {"model.layers.0.proj": GROUP, "model.embed_tokens": GROUP})
    return stored


def test_a_missing_checkpoint_raises_file_not_found_before_touching_any_asset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing.safetensors"):
        te.load_qwen_image_2_1_text_encoder(tmp_path / "missing.safetensors", device=CPU)


def test_the_loader_forces_the_8b_text_dimensions_onto_the_vendored_config(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "te.safetensors")

    te.load_qwen_image_2_1_text_encoder(tmp_path / "te.safetensors", asset_dir=tmp_path, device=CPU)

    text = RECEIVED["config"].text_config
    assert (text.num_hidden_layers, text.hidden_size, text.intermediate_size) == (36, 4096, 12288)
    assert (text.num_attention_heads, text.num_key_value_heads, text.use_cache) == (32, 8, False)
    assert RECEIVED["config"].vision_config.out_hidden_size == 4096
    assert RECEIVED["tokenizer_path"] == str(tmp_path)


def test_without_an_asset_dir_the_bundled_minimax_assets_are_used(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "te.safetensors")

    te.load_qwen_image_2_1_text_encoder(tmp_path / "te.safetensors", device=CPU)

    bundled = Path(RECEIVED["tokenizer_path"])
    assert bundled.parts[-3:] == ("minimax_h3", "assets", "text_encoder") and bundled.is_dir()


def test_the_embedding_table_becomes_an_int8_embedding_holding_the_stored_weights_and_group(
    tmp_path: Path,
) -> None:
    stored = _checkpoint(tmp_path / "te.safetensors")

    encoder = te.load_qwen_image_2_1_text_encoder(tmp_path / "te.safetensors", asset_dir=tmp_path, device=CPU)

    embed = encoder.model.language_model.embed_tokens
    assert isinstance(embed, te.Int8Embedding)
    assert embed.int8_convrot_groupsize.item() == GROUP and embed.embedding_dim == 4096
    assert torch.equal(embed.weight, stored["model.embed_tokens"])
    assert embed.scale_weight.shape == (VOCAB, 1) and embed.scale_weight.min() > 0


def test_linears_load_as_int8_with_their_scale_and_the_lm_head_is_dropped(tmp_path: Path) -> None:
    stored = _checkpoint(tmp_path / "te.safetensors")

    encoder = te.load_qwen_image_2_1_text_encoder(tmp_path / "te.safetensors", asset_dir=tmp_path, device=CPU)

    proj = encoder.model.language_model.layers[0]["proj"]
    assert proj.weight.dtype == torch.int8 and torch.equal(proj.weight, stored["model.layers.0.proj"])
    assert proj.int8_convrot_groupsize.item() == GROUP
    assert not hasattr(encoder.model, "lm_head")


def test_the_final_norm_is_removed_so_the_output_is_the_raw_last_hidden_state(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "te.safetensors")

    encoder = te.load_qwen_image_2_1_text_encoder(tmp_path / "te.safetensors", asset_dir=tmp_path, device=CPU)

    assert isinstance(encoder.model.language_model.norm, nn.Identity)


def test_rotary_modules_are_rebuilt_on_the_target_device_with_the_vision_head_dim(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "te.safetensors")

    encoder = te.load_qwen_image_2_1_text_encoder(tmp_path / "te.safetensors", asset_dir=tmp_path, device=CPU)

    # vision head dim is hidden_size // num_heads = 4, and the rotary table takes half of it
    assert encoder.model.visual.rotary_pos_emb.arg == 2
    assert encoder.model.language_model.rotary_emb.arg is RECEIVED["config"].text_config
    assert encoder.model.language_model.rotary_emb.device_arg == CPU


def test_the_loaded_encoder_is_frozen_eval_and_uses_the_default_prompt_settings(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "te.safetensors")

    encoder = te.load_qwen_image_2_1_text_encoder(tmp_path / "te.safetensors", asset_dir=tmp_path, device=CPU)

    assert isinstance(encoder, te.Qwen3VLTextEncoder) and encoder.tokenizer == "tokenizer"
    assert encoder.system_prompt == te.DEFAULT_SYSTEM_PROMPT and encoder.drop_idx == te.DEFAULT_DROP_IDX
    assert not encoder.model.training
    assert not any(p.requires_grad for p in encoder.model.parameters())


def test_a_checkpoint_that_is_not_int8_convrot_is_rejected(tmp_path: Path) -> None:
    save_file({"model.layers.0.proj.weight": torch.zeros(4, 8)}, str(tmp_path / "plain.safetensors"))

    with pytest.raises(ValueError, match="inconsistent marker"):
        te.load_qwen_image_2_1_text_encoder(tmp_path / "plain.safetensors", asset_dir=tmp_path, device=CPU)
