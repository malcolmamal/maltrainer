"""Plans, member configs and a fake trainer subprocess for the train_combined tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from tests.scripts.helpers import write_lora

D1 = torch.tensor([[1.0, 0, 2, 0], [0, 1, 0, 3]])
U1 = torch.tensor([[1.0, 0], [0, 2], [1, 1]])
D2 = torch.tensor([[0.0, 1, 1, 1], [1, 0, 0, 1]])
U2 = torch.tensor([[1.0, 2], [-1, 0], [2, 1]])


class Trainers:
    """The trainer subprocess: records commands; unless told to fail it writes the member's LoRA."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.codes: list[int] = []
        self.skip_writing = False
        self.lora = {"m": (D1, U1, 2.0)}

    def call(self, command: list[str]) -> int:
        self.commands.append(command)
        code = self.codes.pop(0) if self.codes else 0
        if code == 0 and not self.skip_writing:
            output = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())["output"]
            Path(output["path"]).mkdir(parents=True, exist_ok=True)
            tensors = {"m": (D2, U2, 4.0)} if "second" in command[command.index("--config") + 1] else self.lora
            write_lora(Path(output["path"]) / f"{output['name']}.safetensors", tensors)
        return code


def write_member(tmp_path: Path, name: str) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"output": {"path": str(tmp_path / name), "name": f"lora_{name}"}}))
    return path


def write_plan(tmp_path: Path, **overrides: Any) -> Path:
    (tmp_path / "trainer.py").write_text("")
    config: dict[str, Any] = {
        "trainer": str(tmp_path / "trainer.py"),
        "members": [
            {"config": str(write_member(tmp_path, "first")), "strength": 0.5},
            {"config": str(write_member(tmp_path, "second")), "strength": 2},
        ],
        "output": {"path": str(tmp_path / "soup"), "name": "soup"},
    }
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump({**config, **overrides}))
    return path
