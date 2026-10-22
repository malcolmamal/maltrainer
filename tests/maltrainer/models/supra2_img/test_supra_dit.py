from __future__ import annotations

import pytest
import torch

from maltrainer.models.supra2_img.model import NUM_TOKENS, SupraDiT

CTX = 16


def _tiny(depth: int = 2, tokens: int = 16) -> SupraDiT:
    torch.manual_seed(0)
    model = SupraDiT(latent_ch=4, d_model=32, depth=depth, n_heads=4, ctx_dim=CTX, num_tokens=tokens)
    torch.nn.init.normal_(model.pos_embed, std=0.5)
    return model


def _inputs(batch: int = 2, side: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    return (
        torch.randn(batch, 4, side, side, generator=generator),
        torch.rand(batch, generator=generator),
        torch.randn(batch, 6, CTX, generator=generator),
    )


def test_the_default_model_is_the_104_million_parameter_published_architecture() -> None:
    model = SupraDiT()

    assert round(sum(p.numel() for p in model.parameters()) / 1e6, 1) == 104.1
    assert (len(model.blocks), model.num_tokens, model.pos_embed.shape) == (14, NUM_TOKENS, (1, 256, 576))


@pytest.mark.parametrize("side", [8, 16])
def test_the_velocity_has_the_shape_of_the_latent(side: int) -> None:
    model = _tiny()
    z, t, ctx = _inputs(side=side)

    out = model(z, t, ctx)

    assert out.shape == z.shape and bool(out.isfinite().all())


def test_each_patch_of_a_blockless_model_depends_only_on_its_own_pixels() -> None:
    model = _tiny(depth=0)
    z, t, ctx = _inputs(batch=1)
    changed = z.clone()
    changed[0, 1, 5, 6] += 1.0

    difference = (model(changed, t, ctx) - model(z, t, ctx)).abs().sum(dim=1)[0] > 1e-6

    expected = torch.zeros(8, 8, dtype=torch.bool)
    expected[4:6, 6:8] = True
    assert torch.equal(difference, expected)


def test_samples_in_a_batch_are_processed_independently() -> None:
    model = _tiny()
    z, t, ctx = _inputs(batch=3)

    batched = model(z, t, ctx)

    torch.testing.assert_close(batched[1:2], model(z[1:2], t[1:2], ctx[1:2]), rtol=1e-4, atol=1e-5)


def test_the_timestep_and_the_text_context_both_change_the_prediction() -> None:
    model = _tiny()
    z, t, ctx = _inputs()

    base = model(z, t, ctx)

    assert not torch.allclose(base, model(z, 1 - t, ctx), atol=1e-4)
    assert not torch.allclose(base, model(z, t, ctx + 1), atol=1e-4)


def test_masked_text_tokens_do_not_influence_the_prediction() -> None:
    model = _tiny()
    z, t, ctx = _inputs()
    mask = torch.tensor([[1, 1, 1, 0, 0, 0]] * 2).float()
    changed = ctx.clone()
    changed[:, 3:] = 50.0

    torch.testing.assert_close(model(z, t, ctx, mask), model(z, t, changed, mask), rtol=1e-4, atol=1e-5)


def test_the_learned_positions_are_used_as_they_are_on_the_native_grid() -> None:
    model = _tiny()

    assert model._position_embed(4, 4) is model.pos_embed


def test_other_grids_get_bicubic_resized_positions_in_the_stored_dtype() -> None:
    model = _tiny()
    model.pos_embed.data.fill_(0.25)
    model.to(torch.bfloat16)

    resized = model._position_embed(4, 8)

    assert resized.shape == (1, 32, 32) and resized.dtype == torch.bfloat16
    torch.testing.assert_close(resized.float(), torch.full((1, 32, 32), 0.25))


def test_resized_positions_keep_the_spatial_layout_of_the_learned_grid() -> None:
    model = _tiny()
    ramp = torch.arange(4.0).reshape(1, 4, 1, 1).expand(1, 4, 4, 32)
    model.pos_embed.data = ramp.reshape(1, 16, 32).clone()

    resized = model._position_embed(8, 4).reshape(8, 4, 32)[:, 0, 0]

    assert bool((resized[1:] >= resized[:-1] - 1e-4).all()) and resized[0] < resized[-1]
