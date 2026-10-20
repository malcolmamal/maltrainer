"""scripts/qwen_2_1_bench_score.py: face choice (ArcFace and Haar-crop backends) and the yes/no judge's odds.

``YesNoJudge.__init__`` loads a Hugging Face snapshot of Qwen3-VL; it is not run here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

import qwen_2_1_bench_score as score


class Face:
    def __init__(self, box: tuple[float, float, float, float], vector: list[float]) -> None:
        self.bbox, self.normed_embedding = box, np.array(vector, dtype=np.float32)


def test_with_arcface_the_largest_face_in_each_image_is_embedded_and_an_image_without_one_gives_none() -> None:
    faces = score.FaceEmbedder.__new__(score.FaceEmbedder)
    detections = iter([[Face((0, 0, 10, 10), [1, 0]), Face((0, 0, 20, 20), [0, 3])], []])
    faces.app = SimpleNamespace(get=lambda _array: next(detections))

    out = faces([Image.new("RGB", (30, 30)), Image.new("RGB", (30, 30))])

    assert out[0].tolist() == pytest.approx([0.0, 1.0]) and out[1] is None


def _embedder(clip: Any, root: str | None) -> Any:
    """A FaceEmbedder (typed Any: the clip it takes is a fake)."""
    return score.FaceEmbedder(clip, root, "buffalo_l")


def _fake_insightface(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    made: list[Any] = []

    class FaceAnalysis:
        def __init__(self, **options: Any) -> None:
            self.options: dict[str, Any] = options
            self.prepared: dict[str, Any] | None = None
            made.append(self)

        def prepare(self, **options: Any) -> None:
            self.prepared = options

    package, app = ModuleType("insightface"), ModuleType("insightface.app")
    vars(app)["FaceAnalysis"] = FaceAnalysis
    monkeypatch.setitem(sys.modules, "insightface", package)
    monkeypatch.setitem(sys.modules, "insightface.app", app)
    return made


def test_an_arcface_model_folder_with_an_onnx_file_selects_the_arcface_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    made = _fake_insightface(monkeypatch)
    (tmp_path / "models" / "buffalo_l").mkdir(parents=True)
    (tmp_path / "models" / "buffalo_l" / "det.onnx").touch()

    faces = _embedder(object(), str(tmp_path))

    assert faces.backend == "arcface" and faces.app is made[0]
    assert made[0].options["name"] == "buffalo_l" and made[0].options["root"] == str(tmp_path)
    assert made[0].options["allowed_modules"] == ["detection", "recognition"]
    assert made[0].options["providers"] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert made[0].prepared == {"ctx_id": 0, "det_size": (640, 640)}


class Cascade:
    """OpenCV's Haar cascade, answering with the boxes queued for each call."""

    boxes: list[list[list[int]]] = []  # noqa: RUF012 - set by the test that uses it
    min_sizes: list[tuple[int, int]] = []  # noqa: RUF012 - read back by that test

    def __init__(self, path: str) -> None:
        self.path = path

    def detectMultiScale(self, _gray: Any, **options: Any) -> Any:  # noqa: N802 - OpenCV's name
        Cascade.min_sizes.append(options["minSize"])
        return np.array(Cascade.boxes.pop(0), dtype=int).reshape(-1, 4)


def _fake_cv2(monkeypatch: pytest.MonkeyPatch, boxes: list[list[list[int]]]) -> None:
    Cascade.boxes, Cascade.min_sizes = list(boxes), []
    cv2 = ModuleType("cv2")
    vars(cv2).update(
        CascadeClassifier=Cascade,
        COLOR_RGB2GRAY=0,
        cvtColor=lambda rgb, _code: rgb.mean(axis=2).astype(np.uint8),
        data=SimpleNamespace(haarcascades="cascades/"),
    )
    monkeypatch.setitem(sys.modules, "cv2", cv2)


class RecordingClip:
    def __init__(self) -> None:
        self.crops: list[Image.Image] = []

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        self.crops += images
        return np.array([[float(i + 1), 0.0] for i in range(len(images))])


@pytest.mark.parametrize("root", [None, "missing_root"])
def test_without_arcface_the_largest_haar_face_is_cropped_with_margin_and_embedded_by_clip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], root: str | None
) -> None:
    _fake_cv2(monkeypatch, [[[10, 20, 30, 40], [0, 0, 8, 8]], [], [[0, 0, 200, 100]]])
    clip = RecordingClip()

    faces = _embedder(clip, str(tmp_path / root) if root else None)
    out = faces([Image.new("RGB", (120, 240)), Image.new("RGB", (60, 60)), Image.new("RGB", (100, 100))])

    assert faces.backend == "clip-h-facecrop" and Cascade.min_sizes == [(24, 24)] * 3
    assert [i is None for i in out] == [False, True, False]
    assert out[0].tolist() == [1.0, 0.0] and out[2].tolist() == [2.0, 0.0]
    assert [c.size for c in clip.crops] == [(224, 224)] * 2
    assert ("falling back" in capsys.readouterr().out) == (root is not None)


def test_the_smallest_face_the_cascade_may_find_grows_with_the_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_cv2(monkeypatch, [[]])
    faces = _embedder(RecordingClip(), None)

    faces([Image.new("RGB", (480, 960))])

    assert Cascade.min_sizes == [(40, 40)]


class Processor:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: list[str], **_options: Any) -> Any:
        self.texts += text
        return SimpleNamespace(to=lambda _device: {"input_ids": torch.zeros(1, 3, dtype=torch.long)})


def _judge(yes: float, no: float) -> tuple[Any, Processor]:
    judge = object.__new__(score.YesNoJudge)
    judge.processor, judge.device = Processor(), "cpu"
    judge.yes, judge.no = [1, 2, 3], [4, 5, 6]
    logits = torch.zeros(1, 3, 8)
    logits[0, -1, [1, 2, 3]], logits[0, -1, [4, 5, 6]] = yes, no
    judge.model = lambda **_inputs: SimpleNamespace(logits=logits)
    return judge, judge.processor


@pytest.mark.parametrize(("yes", "no", "odds"), [(1.0, 1.0, 0.5), (3.0, 1.0, 0.8808), (0.0, 4.0, 0.0180)])
def test_the_judge_returns_the_probability_of_yes_against_no_over_all_their_spellings(
    yes: float, no: float, odds: float
) -> None:
    judge, processor = _judge(yes, no)

    answer = judge(Image.new("RGB", (4, 4)), "Is it red?")

    assert answer == pytest.approx(odds, abs=1e-3)
    assert processor.texts == [score.YesNoJudge.TEMPLATE.format(q="Is it red?")]
    assert "Is it red? Answer with Yes or No." in processor.texts[0]
