"""Which model files the LoRA trainer asks for, how it prepares them, and the assistant adapter it can merge."""

from __future__ import annotations

import pytest
import torch

import train_zimage_lora as lora

from .fakes import Fakes
from .tiny import build_transformer
from .world import Workspace, changed, named_weights

Q, K, V = ("layers.0.attention.to_q", "layers.0.attention.to_k", "layers.0.attention.to_v")


def _pair(gen: torch.Generator, name: str, rank: int = 2) -> dict[str, torch.Tensor]:
    return {
        f"{name}.lora_A.weight": torch.randn(rank, 16, generator=gen),
        f"{name}.lora_B.weight": torch.randn(16, rank, generator=gen),
    }


def _adapter() -> dict[str, torch.Tensor]:
    """Pairs for to_q (under "diffusion_model.", no alpha), to_k (under "transformer.", alpha 4) and to_v (bare),
    and what must be skipped: a block that is no Linear, an unknown module, a lone A half and a stray tensor."""
    gen = torch.Generator().manual_seed(5)
    out = {**_pair(gen, f"diffusion_model.{Q}"), **_pair(gen, f"transformer.{K}"), **_pair(gen, V)}
    out[f"transformer.{K}.alpha"] = torch.tensor(4.0)
    out.update(_pair(gen, "layers.0"))
    out.update(_pair(gen, "layers.0.nope"))
    out["layers.0.attention.to_out.0.lora_A.weight"] = torch.randn(2, 16, generator=gen)
    out["stray"] = torch.zeros(1)
    return out


def _delta(adapter: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return adapter[f"{name}.lora_B.weight"] @ adapter[f"{name}.lora_A.weight"]


def test_the_transformer_comes_from_the_model_path_and_the_rest_from_the_extras_path(
    workspace: Workspace, fakes: Fakes
) -> None:
    config = workspace.lora_config(model={"name_or_path": "base/model", "extras_name_or_path": "extras/model"})
    trainer = lora.ZImageLoRATrainer(config)

    trainer.load_model()

    assert [(x.kind, x.path, x.subfolder, x.dtype) for x in fakes.loads] == [
        ("transformer", "base/model", "transformer", torch.float32),
        ("text_encoder", "extras/model", "text_encoder", torch.float32),
        ("tokenizer", "extras/model", "tokenizer", None),
        ("vae", "extras/model", "vae", torch.float32),
    ]


def test_without_an_extras_path_every_part_comes_from_the_model_path(workspace: Workspace, fakes: Fakes) -> None:
    trainer = lora.ZImageLoRATrainer(workspace.lora_config())

    trainer.load_model()

    assert {x.path for x in fakes.loads} == {"tiny/zimage"}


def test_the_vae_stays_float32_while_the_others_take_the_configured_dtype(workspace: Workspace) -> None:
    trainer = lora.ZImageLoRATrainer(workspace.lora_config(dtype="bfloat16"))

    trainer.load_model()

    assert [p.dtype for p in (next(m.parameters()) for m in (trainer.transformer, trainer.text_encoder))] == [
        torch.bfloat16
    ] * 2
    assert next(trainer.vae.parameters()).dtype == torch.float32


def test_every_model_is_frozen_and_the_transformer_checkpoints_its_gradients(workspace: Workspace) -> None:
    trainer = lora.ZImageLoRATrainer(workspace.lora_config())

    trainer.load_model()

    models = (trainer.transformer, trainer.text_encoder, trainer.vae)
    assert not any(p.requires_grad for m in models for p in m.parameters())
    assert trainer.transformer.gradient_checkpointing is True


def test_merging_adds_each_pair_scaled_by_alpha_over_rank_and_skips_the_rest(workspace: Workspace) -> None:
    adapter = _adapter()
    trainer = lora.ZImageLoRATrainer(workspace.lora_config())
    trainer.transformer = build_transformer()
    base = named_weights(build_transformer())

    merged = trainer._merge_lora_into_transformer(adapter)

    after = named_weights(trainer.transformer)
    assert merged == 3
    assert torch.allclose(
        after[f"{Q}.weight"], base[f"{Q}.weight"] + _delta(adapter, f"diffusion_model.{Q}"), atol=1e-6
    )
    assert torch.allclose(
        after[f"{K}.weight"], base[f"{K}.weight"] + _delta(adapter, f"transformer.{K}") * 2, atol=1e-6
    )
    assert torch.allclose(after[f"{V}.weight"], base[f"{V}.weight"] + _delta(adapter, V), atol=1e-6)
    assert changed(base, after) == {f"{Q}.weight", f"{K}.weight", f"{V}.weight"}


def test_the_merge_weight_scales_the_added_delta(workspace: Workspace) -> None:
    adapter = _adapter()
    trainer = lora.ZImageLoRATrainer(workspace.lora_config())
    trainer.transformer = build_transformer()
    base = named_weights(build_transformer())

    trainer._merge_lora_into_transformer(adapter, merge_weight=0.5)

    assert torch.allclose(
        trainer.transformer.state_dict()[f"{V}.weight"], base[f"{V}.weight"] + _delta(adapter, V) * 0.5
    )


def test_an_existing_path_is_used_as_it_is_and_a_hub_path_is_downloaded(workspace: Workspace, fakes: Fakes) -> None:
    trainer = lora.ZImageLoRATrainer(workspace.lora_config())
    local = workspace.root / "a" / "b" / "c.safetensors"
    local.parent.mkdir(parents=True)
    local.touch()

    assert trainer._resolve_hf_path(str(local)) == str(local)
    assert trainer._resolve_hf_path("org/repo/file.safetensors") == "file.safetensors"
    assert trainer._resolve_hf_path("org/repo/sub/dir/file.safetensors") == "sub/dir/file.safetensors"
    assert fakes.downloads == [("org/repo", "file.safetensors"), ("org/repo", "sub/dir/file.safetensors")]


def test_a_path_that_is_neither_local_nor_a_hub_file_is_refused(workspace: Workspace) -> None:
    trainer = lora.ZImageLoRATrainer(workspace.lora_config())

    with pytest.raises(ValueError, match=r"Adapter path org/repo is not a valid local path or hub path\."):
        trainer._resolve_hf_path("org/repo")
