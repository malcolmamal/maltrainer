from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from maltrainer.models.fibo2 import text_encoder as te
from maltrainer.models.flux3 import text_encoder as flux3_te
from maltrainer.models.flux3.text_encoder import TOKENIZER_ASSETS
from maltrainer.models.qwen_image_2_1.text_encoder import Int8Embedding
from tests.maltrainer.models.qwen_image_2_1._tiny import int8_checkpoint

CPU = torch.device("cpu")
VOCAB, GROUP = 6, 4
RECEIVED: dict = {}


class FakeRotary(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.config = config


class FakeTextModel(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        RECEIVED["config"] = config
        self.embed_tokens = nn.Embedding(VOCAB, 4)
        # the encoder hooks layers 5..34, so the stack needs them; only layer 0 carries weights
        self.layers = nn.ModuleList(
            [nn.ModuleDict({"proj": nn.Linear(8, 4, bias=False)})] + [nn.Identity() for _ in range(34)]
        )
        self.norm = nn.LayerNorm(4)
        self.rotary_emb = FakeRotary(None)


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers.models.qwen3_vl.modeling_qwen3_vl as modeling

    RECEIVED.clear()
    monkeypatch.setattr(modeling, "Qwen3VLTextModel", FakeTextModel)
    config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=VOCAB, hidden_size=1, num_hidden_layers=1))
    monkeypatch.setattr(flux3_te.Qwen3VLConfig, "from_pretrained", lambda _path: config)

    def tokenizer(path: str, **_: object) -> str:
        RECEIVED["tokenizer_path"] = path
        return "tokenizer"

    monkeypatch.setattr(te.AutoTokenizer, "from_pretrained", tokenizer)


def _plain(path: Path, prefix: str = "", extra: dict | None = None) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    state = {
        "embed_tokens.weight": torch.randn(VOCAB, 4, generator=generator),
        "layers.0.proj.weight": torch.randn(4, 8, generator=generator),
    }
    noise = {
        "norm.weight": torch.ones(4),
        "layers.35.proj.weight": torch.zeros(4, 8),
        "lm_head.weight": torch.zeros(2, 2),
        "visual.blocks.0.weight": torch.zeros(2, 2),
    }
    save_file({prefix + k: v for k, v in {**state, **noise, **(extra or {})}.items()}, str(path))
    return state


def _load(path: Path, **kwargs: object) -> te.Fibo2TextEncoder:
    return te.load_fibo2_text_encoder(path, CPU, **kwargs)


def test_a_missing_checkpoint_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("gone.safetensors")):
        _load(tmp_path / "gone.safetensors")


@pytest.mark.parametrize("prefix", ["", "model.language_model.", "text_encoders.qwen3vl_4b.transformer."])
def test_a_plain_checkpoint_loads_under_any_prefix_and_extra_layers_norm_head_and_vision_are_ignored(
    tmp_path: Path, prefix: str
) -> None:
    state = _plain(tmp_path / "te.safetensors", prefix)

    encoder = _load(tmp_path / "te.safetensors", dtype=torch.float32)

    model = encoder.model
    assert torch.equal(model.layers[0]["proj"].weight, state["layers.0.proj.weight"])
    assert torch.equal(model.embed_tokens.weight, state["embed_tokens.weight"])
    assert isinstance(model.norm, nn.Identity)


def test_the_config_is_the_4b_text_shape_truncated_to_35_layers(tmp_path: Path) -> None:
    _plain(tmp_path / "te.safetensors")

    encoder = _load(tmp_path / "te.safetensors")

    config = RECEIVED["config"]
    assert (config.hidden_size, config.intermediate_size, config.head_dim) == (2560, 9728, 128)
    assert (config.num_attention_heads, config.num_key_value_heads) == (32, 8)
    assert config.num_hidden_layers == 35 and config.use_cache is False
    assert encoder.model.rotary_emb.config is config
    assert RECEIVED["tokenizer_path"] == str(TOKENIZER_ASSETS)


def test_an_explicit_tokenizer_path_overrides_the_bundled_assets(tmp_path: Path) -> None:
    _plain(tmp_path / "te.safetensors")

    encoder = _load(tmp_path / "te.safetensors", tokenizer_path=tmp_path)

    assert RECEIVED["tokenizer_path"] == str(tmp_path) and encoder.tokenizer == "tokenizer"


def test_float_weights_are_cast_to_the_requested_dtype_and_the_model_is_frozen_eval(tmp_path: Path) -> None:
    _plain(tmp_path / "te.safetensors")

    model = _load(tmp_path / "te.safetensors", dtype=torch.bfloat16).model

    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_a_checkpoint_with_a_tensor_the_text_model_lacks_is_rejected_naming_the_file(tmp_path: Path) -> None:
    _plain(tmp_path / "other.safetensors", extra={"layers.0.mystery.weight": torch.zeros(1)})

    with pytest.raises(ValueError, match=r"other.safetensors does not look like.*unexpected \['layers.0.mystery"):
        _load(tmp_path / "other.safetensors")


def test_a_checkpoint_missing_a_weight_is_rejected_but_missing_rotary_buffers_are_fine(tmp_path: Path) -> None:
    save_file({"embed_tokens.weight": torch.zeros(VOCAB, 4)}, str(tmp_path / "short.safetensors"))

    with pytest.raises(ValueError, match=r"missing \['layers.0.proj.weight'"):
        _load(tmp_path / "short.safetensors")


def _int8(path: Path) -> dict[str, torch.Tensor]:
    state = {
        "model.language_model.layers.0.proj.weight": torch.zeros(4, 8),
        "model.language_model.embed_tokens.weight": torch.zeros(VOCAB, 2560),
        "model.language_model.norm.weight": torch.ones(4),
    }
    names = {"model.language_model.layers.0.proj": GROUP, "model.language_model.embed_tokens": GROUP}
    return int8_checkpoint(path, state, names)


def test_an_int8_convrot_checkpoint_gets_a_rotation_aware_embedding_and_int8_linears(tmp_path: Path) -> None:
    stored = _int8(tmp_path / "q.safetensors")

    model = _load(tmp_path / "q.safetensors", dtype=torch.bfloat16).model

    embed = model.embed_tokens
    assert isinstance(embed, Int8Embedding) and embed.int8_convrot_groupsize.item() == GROUP
    assert torch.equal(embed.weight, stored["model.language_model.embed_tokens"])
    proj = model.layers[0]["proj"]
    assert proj.weight.dtype == torch.int8 and proj.int8_convrot_groupsize.item() == GROUP
    assert torch.equal(proj.weight, stored["model.language_model.layers.0.proj"])
    assert embed.embedding_dim == 2560 and embed.num_embeddings == VOCAB


def test_the_int8_linears_run_through_the_convrot_forward_instead_of_a_plain_linear(tmp_path: Path) -> None:
    _int8(tmp_path / "q.safetensors")

    proj = _load(tmp_path / "q.safetensors", grad_precision="bf16").model.layers[0]["proj"]

    assert proj.int8_convrot_grad_precision == "bf16"
    assert proj.forward.__func__ is not nn.Linear.forward
