from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from maltrainer.models.moss_image.finetune import (
    DRIFT_BLOCKS,
    EXTRACT_PROFILE,
    _container_prefix,
    drift_targets,
    param_group_of,
    param_groups,
    save_full_model,
)
from maltrainer.models.moss_image.model import MOSSImageTransformer

Maker = Callable[..., MOSSImageTransformer]
Writer = Callable[..., Path]


@pytest.mark.parametrize(
    "name, group",
    [
        ("layers.3.proj.q.weight", "attn"),
        ("layers.3.proj.qkv.weight", "attn"),
        ("layers.0.proj.o.weight", "attn"),
        ("layers.0.proj.proj.weight", "attn"),
        ("layers.5.proj.ffn_down.weight", "mlp"),
        ("layers.5.proj.ffn_up_gate.weight", "mlp"),
        ("layers.5.norm1.weight", "norm"),
        ("text_embedder.blocks.0.proj.k.weight", "attn"),
        ("text_embedder.blocks.1.proj.ffn_up.weight", "mlp"),
        ("text_embedder.blocks.1.norm.weight", "norm"),
        ("text_embedder.proj.weight", "other"),
        ("final_layer.linear.weight", "other"),
    ],
)
def test_parameters_are_grouped_by_the_module_family_they_belong_to(name: str, group: str) -> None:
    assert param_group_of(name) == group


def test_unsplit_parameters_form_a_single_group_named_all_in_order() -> None:
    a, b = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(2))

    groups = param_groups([("layers.0.proj.q.weight", a), ("final.weight", b)], split=False)

    assert len(groups) == 1 and groups[0]["name"] == "all"
    assert groups[0]["params"][0] is a and groups[0]["params"][1] is b


def test_split_parameters_form_one_group_per_family_in_first_seen_order() -> None:
    q, ffn, q2, head = (nn.Parameter(torch.zeros(1)) for _ in range(4))
    named = [
        ("layers.0.proj.q.weight", q),
        ("layers.0.proj.ffn_down.weight", ffn),
        ("layers.1.proj.q.weight", q2),
        ("head.weight", head),
    ]

    groups = param_groups(named, split=True)

    assert [g["name"] for g in groups] == ["attn", "mlp", "other"]
    assert groups[0]["params"] == [q, q2]
    assert groups[1]["params"] == [ffn] and groups[2]["params"] == [head]


@pytest.mark.parametrize(
    "keys, prefix",
    [
        (["model.diffusion_model.a", "model.diffusion_model.b"], "model.diffusion_model."),
        (["diffusion_model.a", "diffusion_model.b"], "diffusion_model."),
        (["diffusion_model.a", "b"], ""),
        ([], ""),
    ],
)
def test_the_container_prefix_is_only_recognised_when_every_key_carries_it(keys: list, prefix: str) -> None:
    assert _container_prefix(keys) == prefix


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."])
def test_a_saved_model_keeps_the_layout_keys_order_and_prefix_of_its_source(
    make_model: Maker, write_checkpoint: Writer, tmp_path: Path, prefix: str
) -> None:
    source = make_model()
    layout = write_checkpoint(source, prefix=prefix, extra={"rope.bands": torch.arange(4.0)})
    trained = make_model(seed=1)
    out = tmp_path / "out.safetensors"

    save_full_model(trained, layout, out, metadata={"note": "x"})

    with safe_open(str(layout), framework="pt") as ref, safe_open(str(out), framework="pt") as got:
        assert list(got.keys()) == list(ref.keys())
        assert got.metadata() == {"note": "x"}
        assert torch.equal(got.get_tensor(prefix + "rope.bands"), torch.arange(4.0))
        for key, value in trained.state_dict().items():
            saved = got.get_tensor(prefix + key)
            assert saved.dtype == torch.bfloat16
            assert torch.equal(saved, value.to(torch.bfloat16))
    assert not Path(str(out) + ".tmp").exists()


def test_overrides_replace_the_live_weights_of_the_named_keys_only(
    make_model: Maker, write_checkpoint: Writer, tmp_path: Path
) -> None:
    model = make_model()
    layout = write_checkpoint(model)
    key = next(k for k in model.state_dict() if k.endswith(".weight"))
    replacement = torch.full_like(model.state_dict()[key], 2.0)
    out = tmp_path / "ema.safetensors"

    save_full_model(model, layout, out, overrides={key: replacement}, dtype=torch.float32)

    with safe_open(str(out), framework="pt") as got:
        assert torch.equal(got.get_tensor(key), replacement)
        other = next(k for k in model.state_dict() if k != key)
        assert torch.equal(got.get_tensor(other), model.state_dict()[other])


def test_a_layout_file_missing_a_live_weight_is_refused_and_leaves_no_output(make_model: Maker, tmp_path: Path) -> None:
    model = make_model()
    layout = tmp_path / "partial.safetensors"
    save_file({"only.weight": torch.zeros(1)}, str(layout))
    out = tmp_path / "out.safetensors"

    with pytest.raises(ValueError, match="has no entry for"):
        save_full_model(model, layout, out)

    assert not out.exists()


def test_drift_targets_pick_the_attention_output_and_ffn_down_of_the_tracked_blocks_only() -> None:
    names = []
    for block in (0, 1, 10, 41, 42):
        for leaf in ("proj.o.weight", "proj.ffn_down.weight", "proj.q.weight"):
            names.append(f"layers.{block}.{leaf}")
    names += ["text_embedder.blocks.0.proj.o.weight", "layers.0.proj"]
    params = {n: nn.Parameter(torch.zeros(1)) for n in names}

    targets = drift_targets(list(params.items()))

    assert 10 in DRIFT_BLOCKS and 1 not in DRIFT_BLOCKS
    expected = [
        ("attn", params["layers.0.proj.o.weight"]),
        ("mlp", params["layers.0.proj.ffn_down.weight"]),
        ("attn", params["layers.10.proj.o.weight"]),
        ("mlp", params["layers.10.proj.ffn_down.weight"]),
        ("attn", params["layers.41.proj.o.weight"]),
        ("mlp", params["layers.41.proj.ffn_down.weight"]),
    ]
    assert [(label, id(p)) for label, p in targets] == [(label, id(p)) for label, p in expected]


def test_the_extract_profile_splits_a_fused_qkv_difference_into_q_k_and_v_thirds() -> None:
    diff = torch.arange(12.0).reshape(6, 2)

    parts = EXTRACT_PROFILE.split("layers.4.proj.qkv", diff)

    assert [name for name, _ in parts] == ["layers.4.proj.q", "layers.4.proj.k", "layers.4.proj.v"]
    assert torch.equal(torch.cat([t for _, t in parts]), diff)
    assert parts[1][1].shape == (2, 2)


@pytest.mark.parametrize("name", ["layers.4.proj.o", "text_embedder.blocks.0.proj.qkv", "layers.4.proj.qkv.x"])
def test_other_modules_pass_through_the_qkv_split_untouched(name: str) -> None:
    diff = torch.ones(6, 2)

    assert EXTRACT_PROFILE.split(name, diff) == [(name, diff)]


def test_the_profile_detects_moss_image_only_with_both_marker_tensors() -> None:
    both = ["shared_block_modulation.weight", "layers.0.proj.ffn_up_gate.weight"]

    assert EXTRACT_PROFILE.detect(both) is True
    assert EXTRACT_PROFILE.detect(both[:1]) is False
    assert EXTRACT_PROFILE.detect(both[1:]) is False
    assert EXTRACT_PROFILE.name == "moss_image"
