from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from maltrainer.finetune.extract import extract
from maltrainer.models.moss_image.finetune import EXTRACT_PROFILE, save_full_model
from maltrainer.models.moss_image.loading import load_moss_image_transformer
from maltrainer.models.moss_image.model import MOSSImageTransformer

from .tiny import CPU, json_tensor

Maker = Callable[..., MOSSImageTransformer]
Writer = Callable[..., Path]
EXTRA = {"rope.bands": torch.arange(4.0), "layers.0.comfy_attention.config": json_tensor({"attention": "sage"})}


def _loaded(path: Path, dtype: torch.dtype = torch.float32) -> MOSSImageTransformer:
    return load_moss_image_transformer(path, CPU, dtype)[0]


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "model.diffusion_model."])
def test_a_saved_model_keeps_the_layout_keys_order_and_prefix_of_its_source(
    make_model: Maker, write_checkpoint: Writer, tmp_path: Path, prefix: str
) -> None:
    layout = write_checkpoint(make_model(), prefix=prefix, extra=EXTRA)
    out = tmp_path / "tuned.safetensors"

    save_full_model(_loaded(layout), layout, out, dtype=torch.float32)

    with safe_open(str(layout), "pt") as source, safe_open(str(out), "pt") as saved:
        assert list(saved.keys()) == list(source.keys())
    assert all(key.startswith(prefix) for key in load_file(str(out)))


def test_trained_weights_replace_the_live_ones_and_unused_entries_are_copied_as_stored(
    make_model: Maker, write_checkpoint: Writer, tmp_path: Path
) -> None:
    source = make_model()
    layout = write_checkpoint(source, extra=EXTRA)
    model = _loaded(layout)
    model.layers[0].proj["q"].weight.data.fill_(0.5)
    out = tmp_path / "tuned.safetensors"

    save_full_model(model, layout, out, dtype=torch.float32)

    saved = load_file(str(out))
    assert saved["layers.0.proj.q.weight"].eq(0.5).all()
    assert torch.equal(saved["layers.1.proj.q.weight"], source.state_dict()["layers.1.proj.q.weight"])
    assert saved["rope.bands"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert torch.equal(saved["layers.0.comfy_attention.config"], EXTRA["layers.0.comfy_attention.config"])


def test_weights_are_written_in_the_requested_dtype_and_default_to_bf16(
    make_model: Maker, write_checkpoint: Writer, tmp_path: Path
) -> None:
    layout = write_checkpoint(make_model(), extra=EXTRA)
    out = tmp_path / "tuned.safetensors"

    save_full_model(_loaded(layout), layout, out)

    saved = load_file(str(out))
    assert {v.dtype for k, v in saved.items() if k not in EXTRA} == {torch.bfloat16}
    assert saved["rope.bands"].dtype == torch.float32


def test_overrides_such_as_the_ema_average_win_over_the_live_weights(
    make_model: Maker, write_checkpoint: Writer, tmp_path: Path
) -> None:
    layout = write_checkpoint(make_model(), prefix="diffusion_model.")
    model = _loaded(layout)
    average = torch.full_like(model.state_dict()["layers.1.proj.o.weight"], 0.125)
    out = tmp_path / "ema.safetensors"

    save_full_model(model, layout, out, overrides={"layers.1.proj.o.weight": average}, dtype=torch.float32)

    saved = load_file(str(out))
    assert saved["diffusion_model.layers.1.proj.o.weight"].eq(0.125).all()
    assert torch.equal(saved["diffusion_model.layers.0.proj.o.weight"], model.state_dict()["layers.0.proj.o.weight"])
    assert not torch.equal(model.state_dict()["layers.1.proj.o.weight"], average)


def test_metadata_is_stored_and_the_temp_file_is_gone_once_the_save_lands(
    make_model: Maker, write_checkpoint: Writer, tmp_path: Path
) -> None:
    layout = write_checkpoint(make_model())
    out = tmp_path / "tuned.safetensors"
    out.write_bytes(b"old")

    save_full_model(_loaded(layout), layout, str(out), metadata={"step": "7"})

    with safe_open(str(out), "pt") as saved:
        assert saved.metadata() == {"step": "7"}
    assert sorted(p.name for p in tmp_path.iterdir() if "tuned" in p.name) == ["tuned.safetensors"]


def test_a_model_with_a_weight_the_layout_lacks_is_refused_before_anything_is_written(
    make_model: Maker, write_checkpoint: Writer, tmp_path: Path
) -> None:
    layout = write_checkpoint(make_model(num_layers=1))
    out = tmp_path / "tuned.safetensors"

    with pytest.raises(ValueError, match=r"has no entry for .*layers\.1"):
        save_full_model(make_model(num_layers=2), layout, out)

    assert not out.exists()
    assert not Path(str(out) + ".tmp").exists()


def test_a_saved_finetune_loads_back_as_a_base_and_extracts_to_a_lora_of_what_changed(
    make_model: Maker, write_checkpoint: Writer, tmp_path: Path
) -> None:
    layout = write_checkpoint(make_model(fused_qkv=True))
    model = _loaded(layout)
    model.layers[1].proj["qkv"].weight.data[128:256] += 1.0
    tuned = tmp_path / "tuned.safetensors"
    save_full_model(model, layout, tuned, dtype=torch.float32)

    states, report = extract(layout, tuned, [4], profile=EXTRACT_PROFILE, dtype=torch.float32)

    assert torch.equal(_loaded(tuned).layers[1].proj["qkv"].weight, model.layers[1].proj["qkv"].weight)
    assert sorted(states[4]) == [
        f"diffusion_model.layers.1.proj.k.{s}" for s in ("alpha", "lora_A.weight", "lora_B.weight")
    ]
    up, down = (
        states[4]["diffusion_model.layers.1.proj.k.lora_B.weight"],
        states[4]["diffusion_model.layers.1.proj.k.lora_A.weight"],
    )
    torch.testing.assert_close(up @ down, torch.ones(128, 128), atol=1e-4, rtol=0)
    assert report["profile"] == "moss_image"
