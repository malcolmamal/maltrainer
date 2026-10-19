from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.krea2.vae import Krea2VAE, load_krea2_vae

CPU = torch.device("cpu")


def _source() -> Krea2VAE:
    model = Krea2VAE(dim=4)
    generator = torch.Generator().manual_seed(0)
    for parameter in model.parameters():
        parameter.data = torch.randn(parameter.shape, generator=generator) * 0.2
    return model.eval()


def _file(path: Path, extra: dict | None = None) -> Path:
    state = {k: v.detach().clone() for k, v in _source().state_dict().items()}
    state.update(extra or {})
    save_file(state, str(path))
    return path


def test_the_encoder_loads_from_wans_key_names_and_encodes_like_its_source(tmp_path: Path) -> None:
    images = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(1))

    loaded = load_krea2_vae(_file(tmp_path / "wan.safetensors"), CPU)

    torch.testing.assert_close(loaded.encode(images), _source().encode(images))
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())


def test_decoder_and_temporal_convolution_tensors_in_the_file_are_ignored(tmp_path: Path) -> None:
    extra = {
        "decoder.conv1.weight": torch.zeros(1),
        "encoder.downsamples.2.time_conv.weight": torch.zeros(1),
        "encoder.downsamples.2.time_conv.bias": torch.zeros(1),
    }

    loaded = load_krea2_vae(_file(tmp_path / "wan.safetensors", extra), CPU, dtype=torch.bfloat16)

    assert loaded.conv1.weight.dtype == torch.bfloat16


def test_a_file_missing_encoder_weights_is_refused(tmp_path: Path) -> None:
    state = {k: v for k, v in _source().state_dict().items() if k != "encoder.head.2.bias"}
    path = tmp_path / "wan.safetensors"
    save_file(state, str(path))

    with pytest.raises(ValueError, match=r"is not a Wan 2.1 VAE .*missing \['encoder.head.2.bias'\]"):
        load_krea2_vae(path, CPU)


def test_a_file_without_the_encoder_is_read_at_the_default_width_and_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.safetensors"
    save_file({"unet.weight": torch.zeros(1)}, str(path))

    with pytest.raises(ValueError, match="is not a Wan 2.1 VAE"):
        load_krea2_vae(path, CPU)
