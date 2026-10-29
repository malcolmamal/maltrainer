"""The port against the upstream package (Nikita-Sherstnev/Visage-Image, visage.dit and visage.vae).

The values below were made by that package's own modules on the same name-seeded weights and inputs (tiny_model.fill,
tiny_inputs); on the real files the DiT, the VAE encode and decode and the layer-8 text states matched it exactly
(max abs difference 0.0 in fp32).
"""

from __future__ import annotations

import pytest
import torch

from .tiny_model import LATENT, tiny_inputs, tiny_transformer, tiny_vae

# length -> (sum, mean |v|, the first four values)
DIT = {
    5: (-21.48286, 1.26468, [-0.38102, 0.72461, 0.69484, -0.19136]),
    8: (-30.80083, 1.54568, [-0.52537, 0.41834, 0.38584, -0.0887]),
    2: (-45.65921, 1.871866, [0.18541, 0.58921, 0.43327, 1.03254]),
}


def _check(out: torch.Tensor, expected: tuple[float, float, list[float]]) -> None:
    flat = out.flatten()
    assert flat.double().sum().item() == pytest.approx(expected[0], rel=1e-4)
    assert flat.abs().mean().item() == pytest.approx(expected[1], rel=1e-4)
    assert flat[:4].tolist() == pytest.approx(expected[2], abs=1e-4)


@pytest.mark.parametrize("length", [5, 8, 2])
def test_the_dit_matches_the_upstream_package_for_captions_inside_and_at_a_bucket(length: int) -> None:
    x, t, text = tiny_inputs()
    text = torch.cat([text, text[:, :3]], dim=1) if length == 8 else text[:, :length]

    with torch.no_grad():
        _check(tiny_transformer()(x, t, text), DIT[length])


def test_the_vae_encode_and_decode_match_the_upstream_package() -> None:
    vae = tiny_vae()
    pixels = torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(3))

    latents = vae.encode(pixels)
    _check(latents, (-4.19422, 0.226038, [-0.46517, -0.39522, -0.48931, -0.28288]))
    _check(vae.decode(latents), (1375.07167, 0.447614, [0.23434, 0.98095, 0.88518, 0.29]))
    noise = torch.randn(1, LATENT, 2, 2, generator=torch.Generator().manual_seed(5))
    _check(vae.decode(noise), (1356.22163, 0.441478, [0.5838, 0.97731, 0.16542, 0.02021]))
