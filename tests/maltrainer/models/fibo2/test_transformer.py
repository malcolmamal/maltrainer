from __future__ import annotations

import pytest
import torch

from maltrainer.models.fibo2.config import BUNDLES, Fibo2Config
from maltrainer.models.fibo2.model import Fibo2Transformer

from .tiny_model import CAP, TINY, tiny_inputs, tiny_transformer


def test_the_output_has_the_latents_shape() -> None:
    x, sigma, context = tiny_inputs(h=3, w=2)

    assert tiny_transformer()(x, sigma, context).shape == x.shape


def test_the_same_inputs_give_the_same_output() -> None:
    model = tiny_transformer()
    inputs = tiny_inputs()

    assert torch.equal(model(*inputs), model(*inputs))


def test_sigma_the_latent_and_the_text_all_change_the_prediction() -> None:
    model = tiny_transformer()
    x, sigma, context = tiny_inputs()
    base = model(x, sigma, context)

    assert not torch.allclose(base, model(x, sigma + 0.3, context))
    assert not torch.allclose(base, model(x + 1.0, sigma, context))
    assert not torch.allclose(base, model(x, sigma, context + 1.0))


@pytest.mark.parametrize("bundle", range(BUNDLES))
def test_each_of_the_six_text_bundles_reaches_the_prediction(bundle: int) -> None:
    model = tiny_transformer()
    x, sigma, context = tiny_inputs()
    changed = context.clone()
    changed[..., bundle * CAP : (bundle + 1) * CAP] += 1.0

    assert not torch.allclose(model(x, sigma, context), model(x, sigma, changed))


def test_a_longer_text_gets_more_gist_tokens_and_still_runs() -> None:
    config = Fibo2Config(**{**TINY.__dict__, "max_gist_tokens": 512})
    torch.manual_seed(0)
    model = Fibo2Transformer(config)
    model.compute_dtype = torch.float32
    x, sigma, _ = tiny_inputs(h=1, w=2)

    short = model(x, sigma, torch.randn(1, 10, BUNDLES * CAP))
    long = model(x, sigma, torch.randn(1, 600, BUNDLES * CAP))

    assert short.shape == long.shape == x.shape


def test_only_batch_size_one_is_trained() -> None:
    x, sigma, context = tiny_inputs()

    with pytest.raises(ValueError, match="batch size 1"):
        tiny_transformer()(x.repeat(2, 1, 1, 1), sigma.repeat(2), context.repeat(2, 1, 1))


def test_gradient_checkpointing_changes_neither_the_output_nor_the_gradients() -> None:
    x, sigma, context = tiny_inputs()
    x.requires_grad_(True)
    plain = tiny_transformer()
    checkpointed = tiny_transformer()
    checkpointed.enable_gradient_checkpointing()

    (a,), (b,) = (torch.autograd.grad(m(x, sigma, context).square().sum(), x) for m in (plain, checkpointed))

    assert torch.allclose(a, b, atol=1e-5)
    assert checkpointed.gradient_checkpointing and not plain.gradient_checkpointing


def test_the_state_dict_uses_comfyuis_module_names() -> None:
    keys = set(tiny_transformer().state_dict())

    assert {
        "x_embedder.weight",
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.bias",
        "noise_refiner.0.adaLN_modulation.0.weight",
        "perceiver.latent_queries",
        "perceiver.layers.0.cross_attention.to_kv.weight",
        "perceiver.layers.0.self_attention.to_kv.weight",
        "perceiver.norm_out.weight",
        "layers.0.attention.qkv.weight",
        "layers.0.attention.q_norm.weight",
        "layers.0.feed_forward.w13.weight",
        "layers.1.cross_attn_gate",
        "layers.1.cross_attention.to_q.weight",
        "layers.1.cross_attn_norm2.weight",
        "final_layer.adaLN_modulation.1.weight",
    } <= keys
    assert "layers.0.cross_attention.to_q.weight" not in keys  # only the injection blocks have one


def test_the_compute_dtype_decides_the_activation_dtype() -> None:
    model = tiny_transformer()
    model.compute_dtype = torch.bfloat16
    model.to(torch.bfloat16)
    x, sigma, context = tiny_inputs()

    assert model(x, sigma, context).dtype == torch.bfloat16


def test_the_prediction_is_the_one_comfyuis_fibo2_module_gives_for_the_same_weights() -> None:
    """Golden values taken from comfy/ldm/fibo2/model.py (checkout of 2026-10-24) run on this tiny model's weights;
    the two agreed to 8e-7 at the time. They pin the gist modulation, the RoPE planes and the injection wiring."""
    out = tiny_transformer()(*tiny_inputs())

    assert out.flatten()[:6].tolist() == pytest.approx(
        [-0.48992, -1.02145, -0.02221, -0.56121, -1.61139, -0.35753], abs=1e-4
    )
    assert out.sum().item() == pytest.approx(-13.427, abs=2e-3)
    assert out.abs().mean().item() == pytest.approx(0.667218, abs=1e-5)
