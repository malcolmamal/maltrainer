from __future__ import annotations

import pytest
import torch

from maltrainer.models.krea2 import model as krea2_model
from maltrainer.models.krea2.model import Krea2Config, Krea2Transformer
from tests.maltrainer.models.krea2.tiny_model import TINY, tiny_inputs, tiny_transformer


def _grads(model: Krea2Transformer) -> dict:
    model.zero_grad()
    model(*tiny_inputs()).square().sum().backward()
    return {name: p.grad for name, p in model.named_parameters()}


def test_patchify_folds_each_two_by_two_patch_into_one_token_with_row_and_column_positions() -> None:
    model = tiny_transformer()
    x = torch.arange(16.0).view(1, 1, 4, 4).expand(2, 1, 4, 4)

    tokens, ids, h, w = model.patchify(x)

    assert (h, w) == (2, 2)
    assert tokens[1].tolist() == [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    assert ids[0].tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    assert ids.shape == (2, 4, 3)


def test_patchify_orders_a_tokens_values_channel_first_then_row_then_column() -> None:
    x = torch.zeros(1, 4, 2, 2)
    x[0, 3, 1, 0] = 7.0

    tokens, _, _, _ = tiny_transformer().patchify(x)

    assert tokens[0, 0].nonzero().item() == 3 * 4 + 1 * 2 + 0


def test_unpatchify_inverts_patchify_exactly() -> None:
    model = tiny_transformer()
    x = torch.randn(2, 4, 6, 4)

    tokens, _, h, w = model.patchify(x)

    assert torch.equal(model.unpatchify(tokens, h, w), x)


def test_the_rope_axes_split_the_head_dim_into_a_wide_axis_and_two_equal_ones() -> None:
    assert tiny_transformer().axes_dim == [4, 6, 6]
    wide = Krea2Transformer(
        Krea2Config(features=32, heads=1, kvheads=1, layers=1, multiplier=1, txtdim=16, txtheads=1, txtkvheads=1)
    )
    assert wide.axes_dim == [8, 12, 12]


def test_the_default_config_splits_the_head_dim_into_rope_axes_32_48_48_and_has_12_9b_parameters() -> None:
    with torch.device("meta"):
        model = Krea2Transformer(Krea2Config())

    assert model.axes_dim == [32, 48, 48]
    assert 12.8e9 < sum(p.numel() for p in model.parameters()) < 13.0e9


def test_the_velocity_has_the_shape_of_the_latent_and_the_compute_dtype() -> None:
    x, sigma, context = tiny_inputs()

    out = tiny_transformer()(x, sigma, context)

    assert out.shape == x.shape and out.dtype == torch.float32 and torch.isfinite(out).all()


def test_a_bf16_model_returns_bf16() -> None:
    model = tiny_transformer().to(torch.bfloat16)
    model.compute_dtype = torch.bfloat16

    out = model(*tiny_inputs())

    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()


def test_samples_in_a_batch_do_not_influence_each_other() -> None:
    model = tiny_transformer()
    x, sigma, context = tiny_inputs()

    both = model(x, sigma, context)

    for i in range(2):
        single = model(x[i : i + 1], sigma[i : i + 1], context[i : i + 1])
        torch.testing.assert_close(single, both[i : i + 1], atol=1e-5, rtol=1e-4)


def test_the_output_depends_on_the_noise_level_and_the_prompt_but_not_on_text_length_for_its_shape() -> None:
    model = tiny_transformer()
    x, sigma, context = tiny_inputs()

    base = model(x, sigma, context)

    assert not torch.allclose(base, model(x, sigma.flip(0), context))
    assert not torch.allclose(base, model(x, sigma, context.flip(0)))
    assert model(x, sigma, tiny_inputs(length=5)[2]).shape == base.shape


def test_image_tokens_are_position_aware_so_flipping_the_latent_changes_the_output() -> None:
    model = tiny_transformer()
    x, sigma, context = tiny_inputs()

    assert not torch.allclose(model(x, sigma, context).flip(-1), model(x.flip(-1), sigma, context), atol=1e-4)


@pytest.mark.parametrize("size", [4, 12])
def test_the_forward_pass_accepts_other_resolutions(size: int) -> None:
    x, sigma, context = tiny_inputs(batch=1, h=size, w=size)

    assert tiny_transformer()(x, sigma, context).shape == (1, TINY.channels, size, size)


def test_every_parameter_of_the_dit_receives_a_gradient() -> None:
    grads = _grads(tiny_transformer())

    assert [name for name, grad in grads.items() if grad is None] == []


def test_gradient_checkpointing_does_not_change_the_gradients() -> None:
    plain = _grads(tiny_transformer())
    model = tiny_transformer()
    model.enable_gradient_checkpointing()

    checkpointed = _grads(model)

    assert model.gradient_checkpointing
    for name, grad in plain.items():
        torch.testing.assert_close(checkpointed[name], grad, atol=1e-5, rtol=1e-4)


def test_every_block_is_checkpointed_only_while_gradients_are_on(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def counting(block, *args, use_reentrant):
        assert use_reentrant is False
        calls.append(block)
        return block(*args)

    monkeypatch.setattr(krea2_model, "checkpoint", counting)
    model = tiny_transformer()
    model.enable_gradient_checkpointing()

    with torch.no_grad():
        model(*tiny_inputs())
    assert calls == []
    model(*tiny_inputs())

    assert len(calls) == TINY.layers


def test_the_parameters_carry_comfyuis_module_names() -> None:
    keys = set(tiny_transformer().state_dict())

    assert {
        "first.weight",
        "first.bias",
        "blocks.1.mod.lin",
        "blocks.0.attn.qknorm.qnorm.scale",
        "blocks.0.attn.gate.weight",
        "blocks.0.mlp.down.weight",
        "tmlp.0.weight",
        "tmlp.2.bias",
        "txtfusion.layerwise_blocks.1.attn.wq.weight",
        "txtfusion.projector.weight",
        "txtfusion.refiner_blocks.0.postnorm.scale",
        "txtmlp.0.scale",
        "txtmlp.3.weight",
        "last.norm.scale",
        "last.linear.bias",
        "last.modulation.lin",
        "tproj.1.weight",
    } <= keys
    assert not any(k.startswith("blocks.2") for k in keys)
