"""The text stack on a tiny real Qwen3 (4 layers, cut to 2) with a name-seeded checkpoint; the expected states come from
the full model's ``hidden_states``, not from the loader's own slicing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from maltrainer.models.visage import text_encoder as te

CPU = torch.device("cpu")
CUT = 2
SHIPPED_LAYERS = te.LAYERS  # read before the fixture cuts the model down
TINY = {
    "head_dim": 16,
    "hidden_size": 32,
    "intermediate_size": 64,
    "model_type": "qwen3",
    "num_attention_heads": 2,
    "num_hidden_layers": 4,
    "num_key_value_heads": 1,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
    "vocab_size": 151936,
}


@pytest.fixture(autouse=True)
def tiny_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bundled 28-layer config replaced by a tiny one, and the cut by 2 layers."""
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "config.json").write_text(json.dumps(TINY), encoding="utf-8")
    monkeypatch.setattr(te, "CONFIG_ASSETS", tmp_path / "cfg")
    monkeypatch.setattr(te, "LAYERS", CUT)


def _full_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config.from_pretrained(str(te.CONFIG_ASSETS))).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.3)
    return model


def _checkpoint(path: Path, model: Qwen3ForCausalLM, bare: bool = False) -> Path:
    """The Hugging Face layout (``model.layers.0...``), or without the ``model.`` prefix."""
    state = {
        key.removeprefix("model.") if bare else key: value.detach().clone().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(path))
    return path


def _expected(model: Qwen3ForCausalLM, caption: str, cut: int = CUT) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(str(te.TOKENIZER_ASSETS))
    prefix = tokenizer("Describe the image.", add_special_tokens=False).input_ids
    ids = tokenizer(" " + caption, add_special_tokens=False).input_ids
    with torch.no_grad():
        states = model(input_ids=torch.tensor([prefix + ids]), output_hidden_states=True).hidden_states[cut]
    return states[0, len(prefix) :].to(torch.float8_e4m3fn).float()


@pytest.mark.parametrize("bare", [False, True])
def test_the_states_are_the_full_model_s_layer_2_output_behind_the_prefix_rounded_through_float8(
    tmp_path: Path, bare: bool
) -> None:
    model = _full_model()
    path = _checkpoint(tmp_path / "te.safetensors", model, bare)

    encoder = te.load_visage_text_encoder(path, CPU, torch.float32)
    states = encoder.encode("a woman, red hair", CPU, torch.float32)

    assert torch.equal(states, _expected(model, "a woman, red hair"))
    assert states.shape[1] == TINY["hidden_size"] and states.shape[0] > 3


def test_the_prefix_tokens_are_not_part_of_the_result_but_do_change_it() -> None:
    model = _full_model()
    tokenizer = AutoTokenizer.from_pretrained(str(te.TOKENIZER_ASSETS))
    caption_ids = tokenizer(" hello", add_special_tokens=False).input_ids

    with torch.no_grad():
        bare = model(input_ids=torch.tensor([caption_ids]), output_hidden_states=True).hidden_states[CUT][0]

    assert _expected(model, "hello").shape[0] == len(caption_ids)
    assert not torch.equal(_expected(model, "hello"), bare.to(torch.float8_e4m3fn).float())


def test_the_result_comes_back_in_the_requested_dtype_and_float8_values_are_exact_in_it(tmp_path: Path) -> None:
    model = _full_model()
    encoder = te.load_visage_text_encoder(_checkpoint(tmp_path / "te.safetensors", model), CPU, torch.float32)

    as_bf16 = encoder.encode("a face", CPU, torch.bfloat16)

    assert as_bf16.dtype == torch.bfloat16
    assert torch.equal(as_bf16.float(), _expected(model, "a face"))


def test_the_empty_caption_encodes_as_one_space_token(tmp_path: Path) -> None:
    model = _full_model()
    encoder = te.load_visage_text_encoder(_checkpoint(tmp_path / "te.safetensors", model), CPU, torch.float32)

    states = encoder.encode("", CPU, torch.float32)

    assert states.shape[0] == 1 and torch.equal(states, _expected(model, ""))


def test_a_caption_is_cut_to_the_dits_largest_bucket(tmp_path: Path) -> None:
    encoder = te.load_visage_text_encoder(_checkpoint(tmp_path / "te.safetensors", _full_model()), CPU, torch.float32)

    assert encoder.encode("hair, " * 100, CPU, torch.float32).shape[0] == te.MAX_LENGTH == 128


@pytest.mark.parametrize(
    ("key", "target"),
    [
        ("model.embed_tokens.weight", "embed_tokens.weight"),
        ("layers.1.self_attn.q_proj.weight", "layers.1.self_attn.q_proj.weight"),
        ("model.layers.0.mlp.up_proj.weight", "layers.0.mlp.up_proj.weight"),
        ("model.layers.2.mlp.up_proj.weight", None),
        ("model.layers.27.input_layernorm.weight", None),
        ("model.norm.weight", None),
        ("lm_head.weight", None),
    ],
)
def test_checkpoint_keys_map_onto_the_cut_model_and_the_head_the_norm_and_the_deeper_layers_are_dropped(
    key: str, target: str | None
) -> None:
    assert te.text_key(key) == target


def test_the_model_is_cut_to_the_first_layers_without_a_final_norm_and_is_frozen(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "te.safetensors", _full_model())

    model = te.load_visage_text_encoder(path, CPU, torch.bfloat16).model

    assert len(model.layers) == CUT and model.config.layer_types == ["full_attention"] * CUT
    assert isinstance(model.norm, torch.nn.Identity)
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert not model.training and not any(p.requires_grad for p in model.parameters())


def test_an_explicit_tokenizer_path_overrides_the_bundled_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _checkpoint(tmp_path / "te.safetensors", _full_model())
    seen: list[str] = []
    original = te.AutoTokenizer.from_pretrained

    def record(path: str) -> Any:
        seen.append(path)
        return original(str(te.TOKENIZER_ASSETS))

    monkeypatch.setattr(te.AutoTokenizer, "from_pretrained", record)

    te.load_visage_text_encoder(path, CPU)
    te.load_visage_text_encoder(path, CPU, tokenizer_path=tmp_path)

    assert seen == [str(te.TOKENIZER_ASSETS), str(tmp_path)]


def test_a_missing_checkpoint_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("gone.safetensors")):
        te.load_visage_text_encoder(tmp_path / "gone.safetensors", CPU)


def test_a_checkpoint_with_a_tensor_the_model_lacks_is_rejected_naming_the_file(tmp_path: Path) -> None:
    state = {key: value.detach().clone() for key, value in _full_model().state_dict().items()}
    save_file({**state, "model.layers.0.mystery.weight": torch.zeros(1)}, str(tmp_path / "other.safetensors"))

    with pytest.raises(ValueError, match=r"other.safetensors does not look like.*unexpected \['layers.0.mystery"):
        te.load_visage_text_encoder(tmp_path / "other.safetensors", CPU)


def test_a_checkpoint_missing_a_weight_is_rejected_but_the_rotary_buffers_need_not_be_there(tmp_path: Path) -> None:
    save_file({"model.embed_tokens.weight": torch.zeros(151936, 32)}, str(tmp_path / "short.safetensors"))

    with pytest.raises(ValueError, match=r"missing \['layers.0.self_attn.q_proj.weight'"):
        te.load_visage_text_encoder(tmp_path / "short.safetensors", CPU)


def test_the_bundled_config_is_qwen3_0_6b() -> None:
    config = json.loads((Path(te.__file__).parent / "assets" / "text_encoder" / "config.json").read_text())

    assert (config["hidden_size"], config["intermediate_size"], config["num_hidden_layers"]) == (1024, 3072, 28)
    assert (config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]) == (16, 8, 128)
    assert config["rope_theta"] == 1_000_000 and config["vocab_size"] == 151936
    assert SHIPPED_LAYERS == 8
    assert (te.PREFIX, te.MAX_LENGTH) == ("Describe the image.", 128)
