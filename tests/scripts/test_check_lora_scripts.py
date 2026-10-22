"""scripts/check_my_lora.py and check_good_lora.py: load a LoRA into the Z-Image pipeline and render one test image.

The pipeline is a fake (the real one downloads Tongyi-MAI/Z-Image), so these check the script's own steps.
"""

from __future__ import annotations

import logging
import runpy
from pathlib import Path
from typing import Any

import diffusers
import pytest
import torch

from tests.scripts.zimage_script_fakes import FakePipeline, pipeline_class, restored_logging

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
CASES = pytest.mark.parametrize(
    ("script", "image"), [("check_my_lora.py", "test_my_lora.png"), ("check_good_lora.py", "test_good_lora.png")]
)


def _load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str, fail_lora: bool = False
) -> tuple[Any, type[FakePipeline]]:
    """Run the script's module body as a library (not as the main program), with the fake pipeline."""
    pipeline = pipeline_class(fail_lora=fail_lora)
    monkeypatch.setitem(vars(diffusers), "ZImagePipeline", pipeline)
    monkeypatch.chdir(tmp_path)
    with restored_logging():
        return runpy.run_path(str(SCRIPTS / script), run_name="library"), pipeline


@CASES
def test_the_lora_is_loaded_and_fused_into_the_base_pipeline_and_one_image_is_rendered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, script: str, image: str
) -> None:
    namespace, pipeline = _load(monkeypatch, tmp_path, script)

    with caplog.at_level(logging.INFO):
        namespace["test_load"]("my.safetensors", "a cat")

    (pipe,) = pipeline.instances
    assert (pipe.model_id, pipe.dtype, pipe.device) == ("Tongyi-MAI/Z-Image", torch.bfloat16, "cpu")
    assert pipe.events == ["to cpu", "load_lora my.safetensors", "fuse"]
    assert pipe.calls == [{"args": ("a cat",), "num_inference_steps": 20}]
    assert (tmp_path / image).is_file()
    assert f"Saved {image}" in caplog.text


@CASES
def test_a_lora_that_fails_to_load_is_logged_and_no_image_is_rendered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, script: str, image: str
) -> None:
    namespace, pipeline = _load(monkeypatch, tmp_path, script, fail_lora=True)

    with caplog.at_level(logging.INFO):
        namespace["test_load"]("bad.safetensors", "a cat")

    assert pipeline.instances[0].calls == [] and not (tmp_path / image).exists()
    assert "Failed to load LoRA" in caplog.text and "RuntimeError: bad lora" in caplog.text


@CASES
def test_run_as_a_program_the_script_checks_its_hard_coded_lora_with_the_trigger_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, script: str, image: str
) -> None:
    pipeline = pipeline_class()
    monkeypatch.setitem(vars(diffusers), "ZImagePipeline", pipeline)
    monkeypatch.chdir(tmp_path)

    with restored_logging():
        runpy.run_path(str(SCRIPTS / script), run_name="__main__")

    (pipe,) = pipeline.instances
    assert pipe.events[1].endswith(".safetensors") and pipe.events[2] == "fuse"
    assert pipe.calls[0]["args"] == ("sks woman, photo of a woman, high quality",)
    assert (tmp_path / image).is_file()
