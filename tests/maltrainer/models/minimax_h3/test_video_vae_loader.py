from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.minimax_h3 import video_vae
from tests.maltrainer.models.minimax_h3.h3_video import clip, hf_encoder_key, tiny_video_encoder


@pytest.fixture(autouse=True)
def _tiny_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_vae, "MiniMaxH3VideoEncoderModel", lambda: tiny_video_encoder(seed=99))


def _checkpoint(tmp_path: Path, extra: dict[str, torch.Tensor] | None = None, drop: str | None = None) -> Path:
    source = tiny_video_encoder(seed=1)
    tensors = {hf_encoder_key(k): v.clone() for k, v in source.state_dict().items() if k != drop}
    tensors["decoder.conv_in.weight"] = torch.zeros(1)
    tensors.update(extra or {})
    path = tmp_path / "vae.safetensors"
    save_file(tensors, str(path))
    return path


def test_the_released_key_layout_loads_into_a_half_precision_frozen_encoder(tmp_path: Path) -> None:
    model = video_vae.load_minimax_h3_video_vae(_checkpoint(tmp_path), torch.device("cpu"))

    source = tiny_video_encoder(seed=1)
    assert model.conv_in.weight.dtype == torch.float16
    torch.testing.assert_close(model.conv_in.weight, source.conv_in.weight.half())
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())


def test_the_loaded_encoder_keeps_the_default_latent_statistics(tmp_path: Path) -> None:
    model = video_vae.load_minimax_h3_video_vae(_checkpoint(tmp_path), torch.device("cpu"))

    assert model.latents_mean.tolist() == [0.5, -1.0]


def test_a_checkpoint_missing_encoder_tensors_is_rejected_by_name(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, drop="conv_out.weight")

    with pytest.raises(ValueError, match=r"vae.safetensors did not supply 1 .*conv_out.weight"):
        video_vae.load_minimax_h3_video_vae(path, torch.device("cpu"))


def test_a_checkpoint_with_tensors_the_encoder_lacks_is_rejected(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path, extra={"encoder.surplus.weight": torch.zeros(1)})

    with pytest.raises(ValueError, match=r"mapped 1 tensor.*surplus.weight"):
        video_vae.load_minimax_h3_video_vae(path, torch.device("cpu"))


def test_the_loaded_encoder_encodes(tmp_path: Path) -> None:
    model = video_vae.load_minimax_h3_video_vae(_checkpoint(tmp_path), torch.device("cpu"))

    out = model.encode_image(clip(1, seed=2)[:, :, 0].half().float())

    assert out.shape == (1, 2, 1, 2, 2)
