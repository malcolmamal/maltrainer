from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from maltrainer.models.ltx25.video_vae import (
    Encoder,
    VideoVAEEncoder,
    fold,
    load_video_vae_encoder,
    make_norm,
)

from .tiny_model import randomise

CPU = torch.device("cpu")
BLOCKS = [
    ["res_x", {"num_layers": 1}],
    ["compress_space_res", {"multiplier": 2}],
    ["res_x", {"num_layers": 1}],
    ["compress_time_res", {"multiplier": 2}],
    ["res_x", {"num_layers": 1}],
    ["compress_all_res", {"multiplier": 2}],
    ["res_x", {"num_layers": 1}],
    ["compress_all_res", {"multiplier": 2}],
    ["res_x", {"num_layers": 1}],
]
CONFIG: dict[str, Any] = {
    "patch_size": 4,
    "norm_layer": "pixel_norm",
    "latent_log_var": "uniform",
    "latent_channels": 8,
    "encoder_blocks": BLOCKS,
    "encoder_base_channels": 4,
}


def _vae(config: dict[str, Any] = CONFIG) -> VideoVAEEncoder:
    model = VideoVAEEncoder(config)
    randomise(model)
    return model.eval()


def test_the_latents_are_one_frame_plus_a_frame_per_eight_and_a_32nd_of_the_pixels() -> None:
    latents = _vae().encode(torch.randn(1, 3, 17, 64, 96))

    assert latents.shape == (1, 8, 3, 2, 3)


def test_a_single_frame_gives_a_single_latent_frame() -> None:
    assert _vae().encode(torch.randn(1, 3, 1, 32, 32)).shape == (1, 8, 1, 1, 1)


def test_frames_beyond_the_last_whole_group_of_eight_are_dropped() -> None:
    vae = _vae()
    pixels = torch.randn(1, 3, 12, 32, 32)

    assert vae.encode(pixels).shape[2] == 2
    assert torch.equal(vae.encode(pixels), vae.encode(pixels[:, :, :9]))


def test_the_first_latent_frame_depends_only_on_the_first_frame() -> None:
    vae = _vae()
    pixels = torch.randn(1, 3, 17, 32, 32)
    changed = pixels.clone()
    changed[:, :, 1:] += 1.0

    first, later = vae.encode(pixels), vae.encode(changed)

    assert torch.allclose(first[:, :, 0], later[:, :, 0], atol=1e-5) and not torch.allclose(
        first[:, :, 1:], later[:, :, 1:]
    )


def test_the_latents_are_normalised_with_the_per_channel_statistics() -> None:
    vae = _vae()
    pixels = torch.randn(1, 3, 9, 32, 32)
    stats = vae.per_channel_statistics
    with torch.no_grad():
        stats.get_buffer("mean-of-means").zero_()
        stats.get_buffer("std-of-means").fill_(1.0)
    raw = vae.encode(pixels)
    with torch.no_grad():
        stats.get_buffer("mean-of-means").fill_(0.5)
        stats.get_buffer("std-of-means").fill_(2.0)

    assert torch.allclose(vae.encode(pixels), (raw - 0.5) / 2.0, atol=1e-5)


def test_fold_moves_a_block_of_pixels_into_channels() -> None:
    x = torch.arange(8.0).view(1, 1, 2, 2, 2)

    folded = fold(x, (2, 2, 2))

    assert folded.shape == (1, 8, 1, 1, 1) and folded.flatten().tolist() == list(range(8))


def test_a_group_norm_is_refused_because_its_result_would_depend_on_how_the_clip_is_chunked() -> None:
    with pytest.raises(ValueError, match="unsupported VAE norm_layer 'group_norm'"):
        make_norm("group_norm")


def test_an_unknown_block_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown block: mystery"):
        Encoder({**CONFIG, "encoder_blocks": [["mystery", {}]]})


def test_the_residual_and_plain_compress_blocks_all_build_and_run() -> None:
    config = {
        **CONFIG,
        "encoder_blocks": [
            ["res_x_y", {"multiplier": 2}],
            ["compress_time", {}],
            ["compress_space", {}],
            ["compress_all", {}],
            ["compress_all_x_y", {"multiplier": 2}],
        ],
        "encoder_base_channels": 4,
    }

    assert _vae(config).encode(torch.randn(1, 3, 9, 32, 32)).shape[0] == 1


def _write(path: Path, model: VideoVAEEncoder, config: dict[str, Any] | None = CONFIG, drop: str = "") -> Path:
    state = {k: v.contiguous() for k, v in model.state_dict().items() if not drop or drop not in k}
    state["per_channel_statistics.channel"] = torch.zeros(8)  # the file carries more statistics than are used
    metadata = {"config": json.dumps({"vae": config})} if config is not None else {}
    save_file(state, str(path), metadata=metadata)
    return path


def test_the_loader_reads_the_config_and_the_weights_from_the_file(tmp_path: Path) -> None:
    source = _vae()
    path = _write(tmp_path / "vae.safetensors", source)

    loaded = load_video_vae_encoder(path, CPU)

    pixels = torch.randn(1, 3, 9, 32, 32)
    assert torch.allclose(loaded.encode(pixels), source.encode(pixels), atol=1e-6)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())


def test_the_loader_refuses_a_file_without_config_a_missing_file_and_missing_tensors(tmp_path: Path) -> None:
    source = _vae()

    with pytest.raises(FileNotFoundError, match=r"LTX video VAE not found at .*gone"):
        load_video_vae_encoder(tmp_path / "gone.safetensors", CPU)
    with pytest.raises(ValueError, match="no config metadata"):
        load_video_vae_encoder(_write(tmp_path / "bare.safetensors", source, None), CPU)
    with pytest.raises(ValueError, match=r"lacks encoder tensors.*conv_out"):
        load_video_vae_encoder(_write(tmp_path / "short.safetensors", source, drop="conv_out"), CPU)
    wider = {**CONFIG, "encoder_base_channels": 8}
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_video_vae_encoder(_write(tmp_path / "wide.safetensors", source, wider), CPU)
