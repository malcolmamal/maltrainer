from __future__ import annotations

import pytest
import torch

from maltrainer.models.visage.layers import SinusoidalEmbedding, apply_rope, spiral_rope_tables
from maltrainer.models.visage.model import VisageTransformer

from .tiny_model import CAP, TINY, tiny_inputs, tiny_transformer


def test_the_velocity_has_the_latent_s_shape_and_depends_on_every_input() -> None:
    model = tiny_transformer()
    x, t, text = tiny_inputs()

    out = model(x, t, text)

    assert out.shape == x.shape
    assert not torch.allclose(out, model(x + 1, t, text))
    assert not torch.allclose(out, model(x, t + 0.2, text))
    assert not torch.allclose(out, model(x, t, text + 1))


def test_a_batch_is_the_same_as_its_samples() -> None:
    model = tiny_transformer()
    x, t, text = tiny_inputs()
    other = torch.randn(1, 5, CAP, generator=torch.Generator().manual_seed(1))

    batched = model(torch.cat([x, -x]), torch.cat([t, t + 0.3]), torch.cat([text, other]))

    assert torch.allclose(batched[:1], model(x, t, text), atol=1e-5)
    assert torch.allclose(batched[1:], model(-x, t + 0.3, other), atol=1e-5)


def test_the_text_stream_is_registers_then_the_caption_then_pads_up_to_the_bucket() -> None:
    model = tiny_transformer()
    caption = torch.randn(2, 3, TINY.width)

    stream = model.prepare_text(caption)

    assert stream.shape == (2, TINY.n_text_registers + 4, TINY.width)
    assert torch.equal(stream[:, :2], model.text_registers.expand(2, -1, -1))
    assert torch.equal(stream[:, 2:5], caption)
    assert torch.equal(stream[:, 5], model.text_pad_embed[3].expand(2, -1))  # the pad at position 3, not at 0


def test_a_caption_that_fills_a_bucket_gets_no_pads_and_a_longer_one_takes_the_next_bucket() -> None:
    model = tiny_transformer()

    assert model.prepare_text(torch.randn(1, 4, TINY.width)).shape[1] == 2 + 4
    assert model.prepare_text(torch.randn(1, 5, TINY.width)).shape[1] == 2 + 8


def test_a_caption_longer_than_the_largest_bucket_is_refused() -> None:
    with pytest.raises(ValueError, match="caption of 9 tokens exceeds the largest bucket 8"):
        tiny_transformer().prepare_text(torch.randn(1, 9, TINY.width))


def test_the_pads_follow_the_model_s_dtype() -> None:
    model = tiny_transformer().bfloat16()

    assert model.prepare_text(torch.randn(1, 3, TINY.width, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_the_sinusoid_is_computed_in_fp32_whatever_the_model_s_dtype() -> None:
    embedding = SinusoidalEmbedding(0.01, 100.0, 16).bfloat16()
    t = torch.tensor([[0.3]], dtype=torch.bfloat16)

    out = embedding(t)

    angles = embedding.angular_speeds.float() * t.float()
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.cat([angles.sin(), angles.cos()], dim=-1))


def test_a_bf16_model_runs_on_an_fp32_timestep_and_stays_near_the_fp32_one() -> None:
    reference = tiny_transformer()
    model = tiny_transformer().bfloat16()
    x, t, text = tiny_inputs()

    out = model(x.bfloat16(), t, text.bfloat16())

    assert out.dtype == torch.bfloat16
    similarity = torch.cosine_similarity(out.float().flatten(), reference(x, t, text).flatten(), dim=0)
    assert similarity > 0.9


def test_the_rope_at_the_origin_is_the_identity_and_it_turns_vectors_without_stretching_them() -> None:
    cos, sin = spiral_rope_tables(3, 2, torch.device("cpu"))

    assert cos.shape == sin.shape == (1, 1, 6, 32)
    assert torch.equal(cos[0, 0, 0], torch.ones(32)) and torch.equal(sin[0, 0, 0], torch.zeros(32))
    x = torch.randn(1, 2, 6, 64)
    turned = apply_rope(x, (cos, sin))
    assert torch.allclose(turned.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert not torch.allclose(turned[:, :, 1:], x[:, :, 1:])


def test_the_orientations_turn_a_step_in_x_and_a_step_in_y_differently() -> None:
    cos, _ = spiral_rope_tables(2, 2, torch.device("cpu"), base=10.0, directions=4)

    # pair 0 points along x (angle 0): a step in y (cell 2) leaves it alone, a step in x (cell 1) turns it
    assert cos[0, 0, 2, 0] == 1 and cos[0, 0, 1, 0] != 1
    # pair 2 points along y (angle 90 degrees): the opposite
    assert cos[0, 0, 1, 2].item() == pytest.approx(1, abs=1e-6) and cos[0, 0, 2, 2] != 1


def test_the_freshly_built_model_predicts_zero_velocity() -> None:
    x, t, text = tiny_inputs()

    assert VisageTransformer(TINY)(x, t, text).abs().max() == 0
