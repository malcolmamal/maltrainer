"""The INT8 window rotator: a window comes up as a trainable dense weight and goes back as INT8 codes."""

from __future__ import annotations

import torch
from torch import nn

from maltrainer.models.minimax_h3.lora import MiniMaxH3LoRANetwork
from maltrainer.models.minimax_h3_image.finetune_int8 import decode, encode, group_size

from .int8_model import GROUP, Model, Optimizers, rotator

QKV0, FC0, QKV1, FC1 = (
    "blocks.0.attn.qkv_proj.weight",
    "blocks.0.mlp.fc1.weight",
    "blocks.1.attn.qkv_proj.weight",
    "blocks.1.mlp.fc1.weight",
)


def _linear(model: nn.Module, key: str) -> nn.Linear:
    linear = model.get_submodule(key.removesuffix(".weight"))
    assert isinstance(linear, nn.Linear)
    return linear


def test_a_window_comes_up_as_trainable_dense_weights_and_the_others_stay_int8() -> None:
    model = Model()
    expected = decode(_linear(model, QKV0), torch.float32)
    rot = rotator(model)

    rot.activate(0)

    for key in (QKV0, FC0):
        weight = _linear(model, key).weight
        assert weight.requires_grad and weight.dtype == torch.float32
    assert torch.equal(_linear(model, QKV0).weight.detach(), expected)
    for key in (QKV1, FC1):
        assert _linear(model, key).weight.dtype is torch.int8 and not _linear(model, key).weight.requires_grad
    assert rot.current == 0 and rot.names == ["block 0", "block 1"]


def test_the_window_optimizer_is_made_over_the_windows_weights_by_name() -> None:
    model, optimizers = Model(), Optimizers()
    rot = rotator(model, make_optimizer=optimizers)

    rot.activate(1)

    name, params, optimizer = optimizers.made[0]
    assert name == "block 1" and rot.optimizer is optimizer
    assert [id(p) for p in params] == [id(_linear(model, k).weight) for k in (QKV1, FC1)]


def test_a_trained_window_goes_back_as_int8_codes_of_its_new_value_and_is_remembered() -> None:
    model = Model()
    rot = rotator(model)
    rot.activate(0)
    with torch.no_grad():
        _linear(model, QKV0).weight.add_(0.25)
    trained = _linear(model, QKV0).weight.detach().clone()

    rot.deactivate()

    linear = _linear(model, QKV0)
    assert linear.weight.dtype is torch.int8 and rot.current is None and rot.optimizer is None
    codes, scale = encode(trained, GROUP)
    assert torch.equal(linear.weight, codes) and torch.equal(linear.scale_weight, scale)
    assert torch.equal(rot.master[QKV0], trained)


def test_a_weight_no_window_has_trained_keeps_its_original_codes_exactly() -> None:
    model = Model()
    codes, scale = _linear(model, QKV1).weight.clone(), _linear(model, QKV1).scale_weight.clone()
    rot = rotator(model)

    rot.activate(0)
    rot.deactivate()

    assert torch.equal(_linear(model, QKV1).weight, codes) and torch.equal(_linear(model, QKV1).scale_weight, scale)
    assert rot.master.keys() == {QKV0, FC0}


def test_the_forward_pass_gives_the_same_result_on_codes_on_the_decoded_weight_and_on_codes_again() -> None:
    model = Model()
    linear = _linear(model, QKV0)
    x = torch.randn(3, 8, generator=torch.Generator().manual_seed(0))
    dense = x @ decode(linear, torch.float32).T
    rot = rotator(model)

    frozen = linear(x)
    rot.activate(0)
    live = linear(x)
    rot.deactivate()
    back = linear(x)

    for result in (frozen, live, back):
        torch.testing.assert_close(result, dense, atol=1e-5, rtol=1e-5)


def test_an_adapters_wrapper_on_a_linear_survives_the_swaps() -> None:
    model = Model()
    linear = _linear(model, QKV0)
    layer = nn.Linear(8, 6, bias=False)
    nn.init.constant_(layer.weight, 0.5)
    MiniMaxH3LoRANetwork._wrap_forward(linear, layer.requires_grad_(False))
    x = torch.ones(2, 8)
    rot = rotator(model)
    plain = x @ decode(linear, torch.float32).T

    rot.activate(0)
    on_dense = linear(x)
    rot.deactivate()

    torch.testing.assert_close(on_dense, plain + x @ layer.weight.T, atol=1e-5, rtol=1e-5)
    assert (linear(x) - on_dense).abs().max() < 0.3


def test_weight_reads_the_live_master_or_decoded_value() -> None:
    model = Model()
    rot = rotator(model)
    decoded = decode(_linear(model, QKV1), torch.float32)
    rot.activate(0)
    with torch.no_grad():
        _linear(model, QKV0).weight.fill_(2.0)

    assert torch.equal(rot.weight(QKV0), torch.full((6, 8), 2.0))  # live
    assert torch.equal(rot.weight(QKV1), decoded)  # never trained: decoded
    rot.deactivate()
    assert torch.equal(rot.weight(QKV0), torch.full((6, 8), 2.0))  # remembered


def test_int8_reads_the_codes_a_forward_pass_has_or_encodes_the_live_window() -> None:
    model = Model()
    rot = rotator(model)
    rot.activate(0)
    with torch.no_grad():
        _linear(model, QKV0).weight.fill_(1.0)

    live_codes, live_scale = rot.int8(QKV0)
    frozen_codes, _ = rot.int8(QKV1)

    expected = encode(torch.ones(6, 8), group_size(_linear(model, QKV0)))
    assert torch.equal(live_codes, expected[0]) and torch.equal(live_scale, expected[1])
    assert torch.equal(frozen_codes, _linear(model, QKV1).weight) and frozen_codes.dtype is torch.int8


def test_touched_weights_include_the_live_window() -> None:
    model = Model()
    rot = rotator(model)
    rot.activate(0)
    rot.activate(1)

    assert rot.touched().keys() == {QKV0, FC0, QKV1, FC1}
    assert rotator().touched() == {}
