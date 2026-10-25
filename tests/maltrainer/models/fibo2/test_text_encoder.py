from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from maltrainer.models.fibo2 import text_encoder as te
from maltrainer.models.fibo2.config import BUNDLES
from maltrainer.models.fibo2.prompt import NULL_CAPTION
from maltrainer.models.fibo2.text_encoder import BUNDLE_LAYERS, TAP_LAYERS, Fibo2TextEncoder, unit_rms
from maltrainer.models.flux3.text_encoder import TEMPLATE

CPU = torch.device("cpu")


class AddOne(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + 1.0


class StubModel(nn.Module):
    """35 layers that each add one to a copy of the token id scaled (1, 2), so every tap is known by hand."""

    def __init__(self, as_tuple: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AddOne() for _ in range(35)])
        self.as_tuple = as_tuple
        self.calls: list[dict[str, torch.Tensor]] = []

    def forward(self, **kwargs: torch.Tensor) -> object:
        self.calls.append(kwargs)
        h = kwargs["input_ids"].float()[..., None] * torch.tensor([1.0, 2.0])
        for layer in self.layers:
            h = layer(h)
        return (h,) if self.as_tuple else SimpleNamespace(last_hidden_state=h)


class CharTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is False
        self.texts.append(text)
        return SimpleNamespace(input_ids=[ord(c) for c in text])


def _encoder(**kwargs: object) -> Fibo2TextEncoder:
    return Fibo2TextEncoder(StubModel(**kwargs), CharTokenizer())  # type: ignore[arg-type]


def test_the_fifteen_taps_are_the_distinct_layers_of_the_six_bundles() -> None:
    assert TAP_LAYERS == (5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35)
    assert len(BUNDLE_LAYERS) == BUNDLES
    assert all(len(bundle) == 3 for bundle in BUNDLE_LAYERS)


def test_a_prompt_gives_one_row_per_token_of_six_bundles_of_three_layers() -> None:
    encoder = _encoder()

    out = encoder.encode("a red fox", CPU)

    assert out.shape == (len(TEMPLATE.format("a red fox")), BUNDLES * 3 * 2)
    assert out.dtype == torch.bfloat16


def test_the_prompt_goes_through_the_instruct_template_without_special_tokens_or_padding() -> None:
    encoder = _encoder()

    out = encoder.encode("a red fox", CPU)

    assert encoder.tokenizer.texts == ["<|im_start|>user\na red fox<|im_end|>\n<|im_start|>assistant\n"]
    call = encoder.model.calls[0]
    assert call["input_ids"].shape == (1, out.shape[0])
    assert call["position_ids"][0].tolist() == list(range(out.shape[0]))
    assert call["use_cache"] is False
    assert "attention_mask" not in call


def test_a_json_caption_is_compacted_and_an_empty_prompt_is_the_null_caption() -> None:
    encoder = _encoder()

    encoder.encode('{"short_description": "x", "objects": []}', CPU)
    encoder.encode("", CPU)

    assert encoder.tokenizer.texts == [
        TEMPLATE.format('{"short_description":"x"}'),
        TEMPLATE.format(NULL_CAPTION),
    ]


@pytest.mark.parametrize("as_tuple", [False, True])
def test_each_tap_is_scaled_to_unit_rms_and_laid_out_bundle_by_bundle(as_tuple: bool) -> None:
    out = _encoder(as_tuple=as_tuple).encode("a", CPU, dtype=torch.float32)

    token = float(ord(TEMPLATE.format("a")[0]))
    # layer k adds k to each column of (token, 2 * token)
    expected = []
    for layer in (layer for bundle in BUNDLE_LAYERS for layer in bundle):
        pair = torch.tensor([token + layer, 2 * token + layer])
        expected += (pair * torch.rsqrt(pair.pow(2).mean() + 1e-6)).tolist()
    assert out[0].tolist() == pytest.approx(expected, abs=1e-5)


def test_the_last_tap_is_the_model_output_and_the_others_are_layer_inputs() -> None:
    encoder = _encoder()

    encoder.encode("a", CPU)

    assert sorted(encoder._captured) == list(TAP_LAYERS[:-1])


def test_unit_rms_scales_in_fp32_and_rounds_to_the_requested_dtype() -> None:
    states = torch.tensor([[3.0, 4.0]]).bfloat16()

    out = unit_rms(states, torch.bfloat16)

    assert out.dtype == torch.bfloat16
    assert out.float().pow(2).mean().item() == pytest.approx(1.0, abs=0.02)


def test_the_sequence_is_cut_at_the_maximum_length(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(te, "MAX_SEQUENCE_LENGTH", 30)
    encoder = _encoder()

    out = encoder.encode("a" * 100, CPU)

    assert out.shape[0] == 30


def test_each_call_starts_from_fresh_captures() -> None:
    encoder = _encoder()
    encoder.encode("one", CPU)

    encoder.encode("two", CPU)

    assert sorted(encoder._captured) == list(TAP_LAYERS[:-1])


def test_the_capture_hook_reads_the_hidden_states_from_kwargs_when_called_without_positionals() -> None:
    encoder = _encoder()
    hidden = torch.ones(1, 2, 2, requires_grad=True)

    encoder._capture(5)(None, (), {"hidden_states": hidden})  # type: ignore[arg-type]

    assert not encoder._captured[5].requires_grad
    assert torch.equal(encoder._captured[5], hidden)
