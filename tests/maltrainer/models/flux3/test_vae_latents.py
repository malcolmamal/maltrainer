from __future__ import annotations

import torch
from torch import nn

from maltrainer.models.flux3.vae import BN_EPS, Flux2VAEDecoder, Flux2VAEEncoder

C = 3
MEAN = torch.linspace(-1.0, 1.0, 4 * C)
VAR = torch.linspace(0.5, 2.0, 4 * C)


class Fixed(nn.Module):
    """Stands in for the convolutional trunk: returns a prepared tensor and remembers its input."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output
        self.seen: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen = x
        return self.output


def _identity_conv(conv: nn.Conv2d) -> None:
    conv.weight.data = torch.eye(conv.out_channels).view(conv.out_channels, conv.in_channels, 1, 1)
    conv.bias.data.zero_()


def _encoder(moments: torch.Tensor) -> Flux2VAEEncoder:
    model = Flux2VAEEncoder(ch=32, z_channels=C, embed_dim=C)
    model.encoder = Fixed(moments)
    _identity_conv(model.quant_conv)
    model.bn_running_mean.copy_(MEAN)
    model.bn_running_var.copy_(VAR)
    return model


def _decoder() -> Flux2VAEDecoder:
    model = Flux2VAEDecoder(ch=32, z_channels=C, embed_dim=C)
    model.decoder = nn.Identity()
    _identity_conv(model.post_quant_conv)
    model.bn_running_mean.copy_(MEAN)
    model.bn_running_var.copy_(VAR)
    return model


def _moments() -> torch.Tensor:
    return torch.randn(1, 2 * C, 4, 6, generator=torch.Generator().manual_seed(2))


def test_pixels_are_mapped_from_zero_one_to_minus_one_one_before_the_encoder() -> None:
    model = _encoder(_moments())

    model.encode(torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1).expand(1, 3, 32, 48))

    assert model.encoder.seen[0, :, 0, 0].tolist() == [-1.0, 0.0, 1.0]


def test_the_posterior_mean_is_folded_in_two_by_two_patches_into_channels_and_batch_normed() -> None:
    moments = _moments()
    mean = moments[:, :C]

    latents = _encoder(moments).encode(torch.zeros(1, 3, 32, 48))

    folded = torch.empty(1, 4 * C, 2, 3)
    for c in range(C):
        for dy in range(2):
            for dx in range(2):
                folded[0, c * 4 + dy * 2 + dx] = mean[0, c, dy::2, dx::2]
    expected = (folded - MEAN.view(1, -1, 1, 1)) / torch.sqrt(VAR.view(1, -1, 1, 1) + BN_EPS)
    assert latents.shape == (1, 12, 2, 3)
    torch.testing.assert_close(latents, expected)


def test_the_variance_half_of_the_moments_is_ignored() -> None:
    moments = _moments()
    other = moments.clone()
    other[:, C:] += 10.0

    first = _encoder(moments).encode(torch.zeros(1, 3, 32, 48))
    second = _encoder(other).encode(torch.zeros(1, 3, 32, 48))

    assert torch.equal(first, second)


def test_decoding_undoes_the_batch_norm_and_the_patch_folding_of_encoding() -> None:
    moments = _moments() * 0.25
    latents = _encoder(moments).encode(torch.zeros(1, 3, 32, 48))

    image = _decoder().decode(latents)

    torch.testing.assert_close(image, (moments[:, :C] + 1.0) / 2.0)


def test_decoded_pixels_are_clamped_to_zero_one() -> None:
    decoder = _decoder()
    huge = torch.full((1, 12, 1, 1), 100.0)

    assert decoder.decode(huge).unique().tolist() == [1.0]
    assert decoder.decode(-huge).unique().tolist() == [0.0]


def test_a_bf16_encoder_returns_bf16_latents_and_a_bf16_decoder_float32_pixels() -> None:
    encoder = _encoder(_moments().to(torch.bfloat16)).to(torch.bfloat16)
    decoder = _decoder().to(torch.bfloat16)

    latents = encoder.encode(torch.zeros(1, 3, 32, 48))
    image = decoder.decode(latents)

    assert latents.dtype == torch.bfloat16
    assert image.dtype == torch.float32
    assert image.shape == (1, 3, 4, 6)
