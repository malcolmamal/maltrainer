from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from maltrainer.models.comfy_fp8 import FP8_MAX, dequantize
from maltrainer.models.krea2 import text_encoder as te
from maltrainer.models.krea2.text_encoder import load_krea2_text_encoder

CPU = torch.device("cpu")
TINY_TEXT = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 128,
}
PREFIX = "model.language_model."


@pytest.fixture(autouse=True)
def tiny_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(te, "QWEN3VL_4B_TEXT", TINY_TEXT)
    monkeypatch.setattr(te, "TAP_LAYERS", [1, 2, 3])


def _state() -> dict[str, torch.Tensor]:
    config = Qwen3VLConfig.from_pretrained(str(te.TOKENIZER_ASSETS)).text_config
    for name, value in TINY_TEXT.items():
        setattr(config, name, value)
    config.num_hidden_layers = 3
    torch.manual_seed(0)
    return {k: v.detach().clone() for k, v in Qwen3VLTextModel(config).state_dict().items()}


def _write(path: Path, state: dict[str, torch.Tensor], prefix: str = PREFIX, fp8: list[str] | None = None) -> Path:
    metadata = None
    if fp8 is not None:
        layers = {prefix + name: {"format": "float8_e4m3fn"} for name in fp8}
        metadata = {"_quantization_metadata": json.dumps({"layers": layers})}
    save_file({prefix + k: v.contiguous() for k, v in state.items()}, str(path), metadata=metadata)
    return path


def _fp8_state() -> dict[str, torch.Tensor]:
    state = _state()
    for key in [k for k in state if k.endswith("_proj.weight")]:
        scale = (state[key].abs().amax() / FP8_MAX).float()
        state[key + "_scale"] = scale
        state[key] = (state[key] / scale).to(torch.float8_e4m3fn)
    return state


@pytest.mark.parametrize(
    "prefix", ["", PREFIX, "language_model.model.", "text_encoders.qwen3vl_4b.transformer.", "model."]
)
def test_the_text_model_loads_whatever_prefix_the_file_uses(tmp_path: Path, prefix: str) -> None:
    state = _state()

    encoder = load_krea2_text_encoder(_write(tmp_path / "q.safetensors", state, prefix), CPU, dtype=torch.float32)

    model = encoder.model
    assert len(model.layers) == 3
    assert torch.equal(model.layers[2].self_attn.k_proj.weight, state["layers.2.self_attn.k_proj.weight"])
    assert isinstance(model.norm, torch.nn.Identity)


def test_vision_head_final_norm_and_deeper_layers_in_the_file_are_ignored(tmp_path: Path) -> None:
    state = _state()
    state["visual.blocks.0.weight"] = torch.zeros(1)
    state["lm_head.weight"] = torch.zeros(1)
    state["layers.3.self_attn.q_proj.weight"] = torch.zeros(1)

    encoder = load_krea2_text_encoder(_write(tmp_path / "q.safetensors", state), CPU)

    assert len(encoder.model.layers) == 3


def test_the_loaded_encoder_is_frozen_in_the_requested_dtype_and_encodes_a_prompt(tmp_path: Path) -> None:
    encoder = load_krea2_text_encoder(_write(tmp_path / "q.safetensors", _state()), CPU, dtype=torch.float32)

    out = encoder.encode("a cat", CPU, dtype=torch.float32)

    assert out.shape[1:] == (3, 16) and torch.isfinite(out).all()
    assert not encoder.model.training
    assert not any(p.requires_grad for p in encoder.parameters())
    assert encoder.model.rotary_emb.inv_freq.device.type == "cpu"
    assert {p.dtype for p in encoder.model.parameters()} == {torch.float32}


def test_fp8_layers_are_dequantized_to_the_requested_dtype(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    state = _fp8_state()
    names = [k[: -len(".weight")] for k in state if k.endswith("_proj.weight")]
    path = _write(tmp_path / "q.safetensors", state, fp8=names)

    with caplog.at_level(logging.INFO, logger="maltrainer.models.krea2.text_encoder"):
        encoder = load_krea2_text_encoder(path, CPU, dtype=torch.float32)

    key = "layers.1.mlp.down_proj"
    expected = dequantize(state[f"{key}.weight"], state[f"{key}.weight_scale"], torch.float32)
    assert torch.equal(encoder.model.layers[1].mlp.down_proj.weight, expected)
    assert f"3 layers, {len(names)} fp8 tensors dequantized" in caplog.text


def test_a_tokenizer_path_overrides_the_vendored_tokenizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr(te.AutoTokenizer, "from_pretrained", lambda path, **_: seen.append(path) or "tok")
    path = _write(tmp_path / "q.safetensors", _state())

    encoder = load_krea2_text_encoder(path, CPU, tokenizer_path=tmp_path / "tok")
    load_krea2_text_encoder(path, CPU)

    assert encoder.tokenizer == "tok"
    assert seen == [str(tmp_path / "tok"), str(te.TOKENIZER_ASSETS)]


@pytest.mark.parametrize("damage", ["missing", "unexpected"])
def test_a_file_that_is_not_the_expected_text_model_is_refused(tmp_path: Path, damage: str) -> None:
    state = _state()
    if damage == "missing":
        del state["layers.0.mlp.up_proj.weight"]
    else:
        state["layers.0.bogus.weight"] = torch.zeros(1)

    with pytest.raises(ValueError, match="does not look like a Qwen3-VL-4B text encoder"):
        load_krea2_text_encoder(_write(tmp_path / "q.safetensors", state), CPU)
