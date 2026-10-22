from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from maltrainer.models.moss_image.model import MOSSImageTransformer

Maker = Callable[..., MOSSImageTransformer]


def _inputs(batch: int = 1, h: int = 3, w: int = 2, tokens: int = 4):
    torch.manual_seed(7)
    return torch.randn(batch, 128, h, w), torch.full((batch,), 0.4), torch.randn(batch, tokens, 8)


def _count_block_calls(model: MOSSImageTransformer) -> list[int]:
    calls: list[int] = []
    for index, block in enumerate(model.layers):
        original = block.forward

        def counted(*args, _original=original, _index=index):
            calls.append(_index)
            return _original(*args)

        block.forward = counted
    return calls


def _capture(module: nn.Module) -> dict:
    seen: dict = {}
    module.register_forward_hook(lambda _m, args, out: seen.update(inputs=args[0], output=out))
    return seen


def test_forward_maps_a_latent_to_a_velocity_of_the_same_shape(make_model: Maker) -> None:
    x, sigma, context = _inputs(batch=2, h=3, w=2)

    out = make_model()(x, sigma, context)

    assert out.shape == (2, 128, 3, 2)
    assert out.dtype == torch.float32


def test_forward_runs_in_the_compute_dtype_of_a_bf16_model(make_model: Maker) -> None:
    model = make_model().to(torch.bfloat16)
    model.compute_dtype = torch.bfloat16
    x, sigma, context = _inputs()

    assert model(x, sigma, context).dtype == torch.bfloat16


def test_forward_is_deterministic_and_depends_on_the_latent_sigma_and_text(make_model: Maker) -> None:
    model = make_model()
    x, sigma, context = _inputs()

    out = model(x, sigma, context)

    torch.testing.assert_close(model(x, sigma, context), out)
    assert not torch.allclose(model(x + 1, sigma, context), out)
    assert not torch.allclose(model(x, sigma + 0.3, context), out)
    assert not torch.allclose(model(x, sigma, context + 1), out)


def test_a_single_context_and_a_single_sigma_are_shared_by_every_sample_in_the_batch(make_model: Maker) -> None:
    model = make_model()
    x, _, context = _inputs(batch=2)

    shared = model(x, torch.tensor([0.4]), context[:1])

    explicit = model(x, torch.full((2,), 0.4), context[:1].expand(2, -1, -1))
    torch.testing.assert_close(shared, explicit)


def test_latent_pixels_become_tokens_row_major_and_come_back_to_the_same_place(make_model: Maker) -> None:
    model = make_model()
    embedded, head = _capture(model.video_embedder), _capture(model.final_linear_video)
    x, sigma, context = _inputs(h=3, w=2)

    out = model(x, sigma, context)

    assert embedded["inputs"].shape == (1, 6, 128)
    torch.testing.assert_close(embedded["inputs"][0, 1 * 2 + 0], x[0, :, 1, 0])
    torch.testing.assert_close(out[0, :, 2, 1], head["output"][0, 2 * 2 + 1])


def test_with_zero_modulation_the_blocks_are_skipped_in_effect_and_only_the_head_acts(make_model: Maker) -> None:
    model = make_model()
    x, sigma, context = _inputs()
    skipped = model(x, sigma, context, skip_layers=(0, 1))
    model.shared_block_modulation.weight.data.zero_()
    model.shared_block_modulation.bias.data.zero_()

    gated_off = model(x, sigma, context)

    torch.testing.assert_close(gated_off, skipped, atol=1e-5, rtol=1e-5)


def test_skip_layers_leaves_exactly_those_blocks_out(make_model: Maker) -> None:
    model = make_model(num_layers=3)
    calls = _count_block_calls(model)
    x, sigma, context = _inputs()

    model(x, sigma, context, skip_layers=(1,))

    assert calls == [0, 2]


def test_gradient_checkpointing_recomputes_the_blocks_in_backward_with_the_same_gradients(make_model: Maker) -> None:
    model = make_model()
    calls = _count_block_calls(model)
    x, sigma, context = _inputs()
    model(x, sigma, context).square().sum().backward()
    plain = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad()
    calls.clear()

    model.enable_gradient_checkpointing()
    model(x, sigma, context).square().sum().backward()

    assert calls == [0, 1, 1, 0]
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.grad, plain[name], atol=1e-5, rtol=1e-4)


def test_gradient_checkpointing_does_nothing_without_grad(make_model: Maker) -> None:
    model = make_model()
    model.enable_gradient_checkpointing()
    calls = _count_block_calls(model)
    x, sigma, context = _inputs()

    with torch.no_grad():
        model(x, sigma, context)

    assert calls == [0, 1]


@pytest.mark.parametrize("checkpointing", [False, True])
def test_gradients_reach_only_the_unfrozen_parameters(make_model: Maker, checkpointing: bool) -> None:
    model = make_model()
    model.requires_grad_(False)
    model.layers.requires_grad_(True)
    if checkpointing:
        model.enable_gradient_checkpointing()
    x, sigma, context = _inputs()

    model(x, sigma, context).square().sum().backward()

    got = {name for name, p in model.named_parameters() if p.grad is not None}
    assert got == {name for name, _ in model.layers.named_parameters(prefix="layers")}
