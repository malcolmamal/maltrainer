from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import files

from .conftest import REAL_OUTPUT_ROOT, write_images

Maker = Callable[..., Path]


def test_the_health_endpoint_reports_ok(client: TestClient) -> None:
    response = client.get("/api/health")

    assert (response.status_code, response.json()) == (200, {"status": "ok"})


def test_a_missing_datasets_folder_lists_nothing(client: TestClient, workspace: Path) -> None:
    (workspace / "datasets").rmdir()

    assert client.get("/api/datasets").json() == []


def test_datasets_are_listed_with_their_image_counts_by_name(
    client: TestClient, make_dataset: Maker, workspace: Path
) -> None:
    make_dataset("bob", 3)
    make_dataset("Alice", 2)

    items = client.get("/api/datasets").json()

    assert items == [
        {"name": "Alice", "path": str(workspace / "datasets" / "Alice"), "image_count": 2},
        {"name": "bob", "path": str(workspace / "datasets" / "bob"), "image_count": 3},
    ]


def test_loose_files_next_to_the_datasets_are_not_datasets(client: TestClient, workspace: Path) -> None:
    (workspace / "datasets" / "notes.txt").write_text("not a dataset")

    assert client.get("/api/datasets").json() == []


def test_an_image_count_skips_hidden_subfolders_and_other_file_types(client: TestClient, make_dataset: Maker) -> None:
    folder = make_dataset("alice", 2)
    write_images(folder / "sub", 1)
    write_images(folder / ".ipynb_checkpoints", 5)
    (folder / "img0.txt").write_text("a caption")
    (folder / "SHOUT.JPG").touch()

    (item,) = client.get("/api/datasets").json()

    assert item["image_count"] == 4


def test_an_empty_dataset_folder_is_listed_with_no_images(client: TestClient, workspace: Path) -> None:
    (workspace / "datasets" / "empty").mkdir()

    assert [(i["name"], i["image_count"]) for i in client.get("/api/datasets").json()] == [("empty", 0)]


def test_the_datasets_folder_defaults_to_datasets_under_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(files.DATASETS_ENV)
    monkeypatch.chdir(tmp_path)

    assert files.datasets_root() == (tmp_path / "datasets").resolve()


def test_the_models_folder_is_unset_without_its_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(files.COMFYUI_MODELS_ENV)

    assert files.comfyui_models_root() is None


def test_a_scan_is_reused_until_it_is_older_than_the_cache_window(make_dataset: Maker) -> None:
    folder = make_dataset("alice", 2)
    assert files.scan_images(folder)[0] == 2
    write_images(folder / "more", 3)

    assert files.scan_images(folder)[0] == 2

    stamp, result = files._scan_cache[str(folder)]
    files._scan_cache[str(folder)] = (stamp - files.SCAN_CACHE_SECONDS - 1, result)

    assert files.scan_images(folder)[0] == 5


@pytest.mark.parametrize(("size", "non_square"), [((100, 100), 0), ((100, 103), 0), ((100, 104), 1), ((104, 100), 1)])
def test_an_image_is_non_square_past_a_three_percent_difference(
    workspace: Path, size: tuple[int, int], non_square: int
) -> None:
    folder = write_images(workspace / "datasets" / "alice", 1, size)

    assert files.scan_images(folder) == (1, non_square, size if non_square else None)


def test_an_unreadable_image_counts_but_is_not_judged_non_square(workspace: Path) -> None:
    folder = write_images(workspace / "datasets" / "alice", 1, (20, 40))
    (folder / "broken.png").write_bytes(b"not an image")

    assert files.scan_images(folder) == (2, 1, (20, 40))


def test_a_scan_ignores_files_that_are_not_images(workspace: Path) -> None:
    folder = write_images(workspace / "datasets" / "alice", 2)
    (folder / "img0.txt").write_text("a caption")

    assert files.scan_images(folder) == (2, 0, None)


def test_runs_are_written_below_an_output_folder_in_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    assert REAL_OUTPUT_ROOT() == (tmp_path / "output").resolve()
