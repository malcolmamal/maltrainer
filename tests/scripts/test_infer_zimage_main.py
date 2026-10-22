"""scripts/infer_zimage.py on the command line: pipeline loading, LoRA and embedding, seeds and output names."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from tests.scripts.zimage_script_fakes import (
    WORDS,
    FakePipeline,
    Loader,
    loaded_infer,
    pipeline_class,
    tiny_text_encoder,
    tiny_tokenizer,
)

STAMP = "20260101_120000"


@pytest.fixture
def pipeline() -> type[FakePipeline]:
    return pipeline_class()


@pytest.fixture
def infer(monkeypatch: pytest.MonkeyPatch, pipeline: type[FakePipeline], tmp_path: Path) -> Iterator[ModuleType]:
    monkeypatch.chdir(tmp_path)
    with loaded_infer(monkeypatch, pipeline) as module:
        monkeypatch.setattr(module, "datetime", SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda _: STAMP)))
        yield module


def _run(infer: ModuleType, monkeypatch: pytest.MonkeyPatch, *argv: object) -> FakePipeline:
    monkeypatch.setattr("sys.argv", ["infer_zimage.py", *map(str, argv)])
    infer.main()
    return infer.ZImagePipeline.instances[-1]


def _seeds(pipe: FakePipeline) -> list[int]:
    return [call["generator"].initial_seed() for call in pipe.calls]


def test_the_defaults_load_the_base_model_in_bfloat16_and_save_one_stamped_image(
    infer: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipe = _run(infer, monkeypatch, "--prompt", "a cat", "--seed", 5)

    assert (pipe.model_id, pipe.dtype, pipe.device, pipe.events) == (
        "Tongyi-MAI/Z-Image",
        torch.bfloat16,
        "cpu",
        ["to cpu"],
    )
    assert pipe.calls == [
        {
            "args": (),
            "prompt": "a cat",
            "guidance_scale": 3.5,
            "num_inference_steps": 20,
            "width": 1024,
            "height": 1024,
            "generator": pipe.calls[0]["generator"],
        }
    ]
    assert _seeds(pipe) == [5]
    assert [p.name for p in tmp_path.iterdir()] == [f"output_{STAMP}.png"]


def test_model_guidance_steps_and_size_are_passed_on_and_the_output_keeps_its_folder_and_extension(
    infer: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "out").mkdir()
    flags = ["--model_id", "me/model", "--guidance_scale", "1.5", "--num_inference_steps", "4"]
    flags += ["--width", "64", "--height", "32", "--output_path", "out/pic.jpg", "--seed", "1"]

    pipe = _run(infer, monkeypatch, "--prompt", "x", *flags)

    call = pipe.calls[0]
    assert pipe.model_id == "me/model"
    assert (call["guidance_scale"], call["num_inference_steps"], call["width"], call["height"]) == (1.5, 4, 64, 32)
    assert [p.name for p in (tmp_path / "out").iterdir()] == [f"pic_{STAMP}.jpg"]


def test_several_images_use_consecutive_seeds_and_numbered_names_with_png_as_the_default_extension(
    infer: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipe = _run(infer, monkeypatch, "--prompt", "x", "--num_images", 3, "--seed", 10, "--output_path", "gen")

    assert _seeds(pipe) == [10, 11, 12]
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"gen_{STAMP}_{i}.png" for i in (1, 2, 3)]


def test_without_a_seed_a_random_one_is_drawn_once_and_counted_up(
    infer: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipe = _run(infer, monkeypatch, "--prompt", "x", "--num_images", 2)

    first, second = _seeds(pipe)
    assert 0 <= first < 2**32 and second == first + 1


def test_a_lora_is_loaded_and_fused_before_generating(infer: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    pipe = _run(infer, monkeypatch, "--prompt", "x", "--seed", 1, "--lora_path", "my.safetensors")

    assert pipe.events == ["to cpu", "load_lora my.safetensors", "fuse"]


def test_an_embedding_replaces_the_tokenizer_and_text_encoder_before_generating(
    infer: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    save_file({"embedding": torch.ones(1, 8)}, str(tmp_path / "e.safetensors"))
    for name, build in (("Qwen2Tokenizer", tiny_tokenizer), ("Qwen3ForCausalLM", tiny_text_encoder)):
        monkeypatch.setattr(infer, name, Loader(build, []))
    seen: dict[str, Any] = {}
    real = infer.load_textual_inversion_embedding

    def spy(pipe: Any, path: str, token: str, model_id: str) -> None:
        real(pipe, path, token, model_id)
        seen.update(tokens=len(pipe.tokenizer), token=pipe.tokenizer.convert_tokens_to_ids(token), path=path)

    monkeypatch.setattr(infer, "load_textual_inversion_embedding", spy)

    _run(
        infer,
        monkeypatch,
        "--prompt",
        "x",
        "--seed",
        1,
        "--embedding_path",
        tmp_path / "e.safetensors",
        "--token_string",
        "<cat>",
    )

    assert seen == {"tokens": len(WORDS) + 1, "token": len(WORDS), "path": str(tmp_path / "e.safetensors")}


def test_the_prompt_is_required(
    infer: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["infer_zimage.py"])

    with pytest.raises(SystemExit) as stop:
        infer.main()

    assert stop.value.code == 2 and "the following arguments are required: --prompt" in capsys.readouterr().err


def test_an_output_folder_that_does_not_exist_yet_is_created(
    infer: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run(infer, monkeypatch, "--prompt", "x", "--seed", 1, "--output_path", tmp_path / "new" / "deep" / "pic.png")

    assert [p.name for p in (tmp_path / "new" / "deep").iterdir()] == [f"pic_{STAMP}.png"]


def test_the_embedding_is_loaded_for_the_model_that_was_asked_for(
    infer: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    save_file({"embedding": torch.ones(1, 8)}, str(tmp_path / "e.safetensors"))
    asked: list[tuple[str, str]] = []
    for name, build in (("Qwen2Tokenizer", tiny_tokenizer), ("Qwen3ForCausalLM", tiny_text_encoder)):
        monkeypatch.setattr(infer, name, Loader(build, asked))

    _run(
        infer, monkeypatch, "--prompt", "x", "--seed", 1, "--model_id", "me/other",
        "--embedding_path", tmp_path / "e.safetensors",
    )  # fmt: skip

    assert {model for model, _ in asked} == {"me/other"}
