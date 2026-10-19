from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import resize_lora
from tests.scripts.helpers import KOHYA, delta, write_lora

# W = 2 * up @ down has singular values 20, 10, 0.2, 0.02 along the first four axes.
SPECTRUM = torch.tensor([10.0, 5.0, 0.1, 0.01])
UP = torch.eye(5, 4) * SPECTRUM
DOWN = torch.eye(4, 6)
ALPHA = 8.0  # rank 4 -> factor 2


def reconstruct(down: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return up.double() @ down.double()


def test_rank_cap_keeps_the_dominant_directions():
    down, up, keep, error = resize_lora.resize_module(DOWN, UP, ALPHA, max_rank=2, frac=1.0, device="cpu")
    assert keep == 2
    assert down.shape == (2, 6) and up.shape == (5, 2)
    full = delta(DOWN, UP, ALPHA)
    assert torch.allclose(reconstruct(down, up), full[:, :6] * (torch.arange(5) < 2)[:, None], atol=1e-4)
    assert error == pytest.approx(((0.2**2 + 0.02**2) / (20**2 + 10**2 + 0.2**2 + 0.02**2)) ** 0.5, rel=1e-4)


def test_error_is_the_eckart_young_optimum_for_a_random_lora():
    generator = torch.Generator().manual_seed(0)
    down, up = torch.randn(6, 9, generator=generator), torch.randn(7, 6, generator=generator)
    full = delta(down, up, 3.0)  # alpha 3 / rank 6
    s = torch.linalg.svdvals(full)
    for cap in (1, 3, 6):
        new_down, new_up, keep, error = resize_lora.resize_module(down, up, 3.0, cap, 1.0, "cpu")
        assert keep == cap
        actual = (full - reconstruct(new_down, new_up)).norm() / full.norm()
        assert actual.item() == pytest.approx(error, rel=1e-3, abs=1e-5)
        assert error == pytest.approx(((s[cap:] ** 2).sum() / (s**2).sum()).sqrt().item(), rel=1e-3, abs=1e-5)


def test_frac_picks_the_rank_when_under_the_cap():
    # 20^2/500.04 = 0.80 < 0.9; (20^2 + 10^2)/500.04 = 0.99992 >= 0.9
    assert resize_lora.resize_module(DOWN, UP, ALPHA, 4, 0.9, "cpu")[2] == 2
    assert resize_lora.resize_module(DOWN, UP, ALPHA, 4, 0.7, "cpu")[2] == 1
    assert resize_lora.resize_module(DOWN, UP, ALPHA, 4, 0.9999999, "cpu")[2] == 4


def test_cap_larger_than_rank_and_zero_module():
    assert resize_lora.resize_module(DOWN, UP, ALPHA, 64, 1.0, "cpu")[2] == 4
    down, up, keep, error = resize_lora.resize_module(DOWN, torch.zeros(5, 4), ALPHA, 3, 0.999, "cpu")
    assert (keep, error) == (1, 0.0)
    assert not reconstruct(down, up).any()


def test_resize_file_writes_scale_one_loras_with_metadata(tmp_path):
    src = write_lora(tmp_path / "a.safetensors", {"m": (DOWN, UP, ALPHA)}, layout=KOHYA, metadata={"ss_x": "keep"})
    dst = tmp_path / "out" / "a_r2.safetensors"
    resize_lora.resize_file(src, dst, 2, 1.0, torch.float32, "cpu")

    state = load_file(str(dst))
    assert sorted(state) == ["m.alpha", "m.lora_down.weight", "m.lora_up.weight"]  # layout kept
    assert float(state["m.alpha"]) == 2.0
    approx = reconstruct(state["m.lora_down.weight"], state["m.lora_up.weight"])
    expected = delta(DOWN, UP, ALPHA) * (torch.arange(5) < 2)[:, None]
    assert torch.allclose(approx, expected[:, :6], atol=1e-4)
    with safe_open(str(dst), framework="pt") as handle:
        meta = handle.metadata()
    assert (meta["ss_x"], meta["ss_network_dim"], meta["ss_network_alpha"]) == ("keep", "2", "2")
    recipe = json.loads(meta["maltrainer_resize"])
    assert recipe["original_ranks"] == [4] and recipe["max_rank"] == 2
    assert recipe["source"]["file"] == "a.safetensors"


def run(monkeypatch, *argv) -> int:
    monkeypatch.setattr("sys.argv", ["resize_lora.py", *map(str, argv), "--device", "cpu", "--dtype", "float32"])
    return resize_lora.main()


def test_main_default_name_and_rank(tmp_path, monkeypatch):
    src = write_lora(tmp_path / "a.safetensors", {"m": (DOWN, UP, ALPHA)})
    assert run(monkeypatch, src, "--rank", "3", "--frac", "1") == 0
    state = load_file(str(tmp_path / "a_resized.safetensors"))
    assert state["m.lora_A.weight"].shape == (3, 6)
    assert state["m.lora_B.weight"].shape == (5, 3)
    assert load_file(str(src))["m.lora_A.weight"].shape == (4, 6)  # input untouched


def test_main_bad_file_returns_1_and_good_one_still_written(tmp_path, monkeypatch, capsys):
    good = write_lora(tmp_path / "good.safetensors", {"m": (DOWN, UP, ALPHA)})
    bad = tmp_path / "bad.safetensors"
    save_file({"w": torch.ones(2, 2)}, str(bad))
    assert run(monkeypatch, bad, good, "--suffix", "_s") == 1
    assert "no lora_A" in capsys.readouterr().err
    assert (tmp_path / "good_s.safetensors").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--rank", "0"], "--rank must be at least 1"),
        (["--frac", "0"], "--frac must be in"),
        (["--frac", "1.5"], "--frac must be in"),
        (["--suffix", ""], "would overwrite the input"),
        (["--output", "x.safetensors", "--second"], "single input"),
        (["--output", "{dir}/exists.safetensors"], "pass --force"),
    ],
)
def test_main_argument_errors(tmp_path, monkeypatch, capsys, extra, message):
    src = write_lora(tmp_path / "a.safetensors", {"m": (DOWN, UP, ALPHA)})
    (tmp_path / "exists.safetensors").write_bytes(b"x")
    inputs = [src, src] if "--second" in extra else [src]
    extra = [e.format(dir=tmp_path) for e in extra if e != "--second"]
    with pytest.raises(SystemExit) as raised:
        run(monkeypatch, *inputs, *extra)
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_main_missing_input(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        run(monkeypatch, tmp_path / "nope.safetensors")
    assert "not found" in capsys.readouterr().err
