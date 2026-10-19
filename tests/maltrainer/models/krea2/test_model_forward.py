"""Krea 2 attention, patching and the full forward pass at toy sizes."""

from __future__ import annotations

import pytest
import torch

from maltrainer.models.krea2.model import Attention, Krea2Config, Krea2Transformer

from .tiny import CONFIG, inputs, tiny_model


def test_grouped_query_attention_equals_full_heads_with_repeated_kv_weights():
    torch.manual_seed(0)
    grouped = Attention(16, heads=4, kvheads=2)
    full = Attention(16, heads=4)
    shared = {k: v for k, v in grouped.state_dict().items() if not k.startswith(("wk", "wv"))}
    full.load_state_dict(shared, strict=False)
    for name in ("wk", "wv"):
        weight = getattr(grouped, name).weight.data.view(2, 4, 16)  # (kv head, head dim, in)
        getattr(full, name).weight.data = weight.repeat_interleave(2, dim=0).reshape(16, 16)
    x = torch.randn(2, 5, 16)
    assert grouped.wk.weight.shape == (8, 16)
    assert torch.allclose(grouped(x), full(x), atol=1e-6)


def test_attention_output_is_gated_by_a_sigmoid_of_the_input_projection():
    torch.manual_seed(0)
    attn = Attention(8, heads=2)
    attn.wo.weight.data = torch.eye(8)
    x = torch.randn(1, 4, 8).abs()  # positive input: a positive gate weight saturates the sigmoid to 1
    attn.gate.weight.data = torch.full((8, 8), 1e3)
    open_gate = attn(x)
    attn.gate.weight.data.zero_()  # sigmoid(0) = 1/2
    assert torch.allclose(attn(x), open_gate / 2, atol=1e-6)
    attn.gate.weight.data = torch.full((8, 8), -1e3)
    assert torch.allclose(attn(x), torch.zeros(1, 4, 8), atol=1e-6)


def test_attention_mask_hides_masked_keys():
    torch.manual_seed(0)
    attn = Attention(8, heads=2)
    x = torch.randn(1, 4, 8)
    mask = torch.tensor([True, True, False, False]).expand(4, 4)
    perturbed = x.clone()
    perturbed[:, 2:] += 5.0
    assert torch.allclose(attn(x, mask=mask)[:, :2], attn(perturbed, mask=mask)[:, :2], atol=1e-5)


def test_patchify_puts_each_2x2_patch_in_a_token_in_channel_major_order():
    model = tiny_model()
    x = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)  # 4 channels, 4x4
    tokens, ids, h, w = model.patchify(x)
    assert (h, w) == (2, 2) and tokens.shape == (1, 4, 16)
    # token 1 = patch (row 0, col 1): per channel the 2x2 block at rows 0-1, cols 2-3
    expected = torch.cat([x[0, c, 0:2, 2:4].flatten() for c in range(4)])
    assert tokens[0, 1].tolist() == expected.tolist()
    assert ids[0].tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]


def test_unpatchify_inverts_patchify():
    model = tiny_model()
    x = torch.randn(3, 4, 6, 10)
    tokens, _, h, w = model.patchify(x)
    assert torch.equal(model.unpatchify(tokens, h, w), x)


def test_default_config_head_dim_splits_into_rope_axes_32_48_48_and_is_12_9b_params():
    with torch.device("meta"):
        model = Krea2Transformer(Krea2Config())
    assert model.axes_dim == [32, 48, 48]
    assert 12.8e9 < sum(p.numel() for p in model.parameters()) < 13.0e9


def test_tiny_rope_axes_cover_the_head_dim():
    assert tiny_model().axes_dim == [4, 6, 6]


def test_forward_returns_the_velocity_in_latent_shape_and_dtype():
    x, sigma, context = inputs(size=8)
    out = tiny_model()(x, sigma, context)
    assert out.shape == x.shape and out.dtype == torch.float32 and torch.isfinite(out).all()


def test_forward_runs_in_the_compute_dtype():
    model = tiny_model().to(torch.bfloat16)
    model.compute_dtype = torch.bfloat16
    x, sigma, context = inputs(batch=1)
    assert model(x, sigma, context).dtype == torch.bfloat16


def test_forward_depends_on_sigma_and_on_the_prompt():
    model = tiny_model()
    x, sigma, context = inputs(batch=1)
    base = model(x, sigma, context)
    assert not torch.allclose(base, model(x, sigma * 0.5, context), atol=1e-4)
    assert not torch.allclose(base, model(x, sigma, context + 1.0), atol=1e-4)


def test_forward_samples_do_not_see_each_other():
    model = tiny_model()
    x, sigma, context = inputs(batch=2)
    both = model(x, sigma, context)
    alone = model(x[1:], sigma[1:], context[1:])
    assert torch.allclose(both[1:], alone, atol=1e-5)


def test_image_tokens_are_position_aware_so_flipping_the_latent_changes_the_output():
    model = tiny_model()
    x, sigma, context = inputs(batch=1)
    assert not torch.allclose(model(x, sigma, context).flip(-1), model(x.flip(-1), sigma, context), atol=1e-4)


def test_gradient_checkpointing_gives_the_same_loss_and_gradients():
    model = tiny_model()
    x, sigma, context = inputs(batch=1)

    def grads():
        model.zero_grad()
        model(x, sigma, context).square().sum().backward()
        return {n: p.grad.clone() for n, p in model.named_parameters()}

    plain = grads()
    model.enable_gradient_checkpointing()
    assert model.gradient_checkpointing
    checked = grads()
    assert plain.keys() == checked.keys()
    for name in plain:
        assert torch.allclose(plain[name], checked[name], atol=1e-4), name


def test_checkpointing_is_skipped_under_no_grad():
    model = tiny_model()
    model.enable_gradient_checkpointing()
    x, sigma, context = inputs(batch=1)
    with torch.no_grad():
        out = model(x, sigma, context)
    assert out.shape == x.shape


@pytest.mark.parametrize("size", [4, 12])
def test_forward_accepts_other_resolutions(size):
    x, sigma, context = inputs(batch=1, size=size)
    assert tiny_model()(x, sigma, context).shape == (1, CONFIG.channels, size, size)
