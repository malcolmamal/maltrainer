#!/usr/bin/env python3
"""Train one dataset with several optimizers in turn, then merge the LoRAs into one file.

A combined run is a "LoRA soup": each member config trains its own LoRA (the same model and
dataset, a different optimizer), and the results are merged exactly with ``merge_loras.py`` at a
strength per member, so the soup applies what stacking them at those strengths does.

    python scripts/train_combined.py --config config/moss_image_template-combined__feliciaday.yaml

The config lists the member configs (already filled in, as for a single run), their strengths and
where the merge goes:

    trainer: train_moss_image_lora.py          # the trainer script every member runs with
    members:
      - config: config/moss_image_template-adamw8bit__feliciaday.yaml
        strength: 0.7
      - config: config/moss_image_template__feliciaday.yaml
        strength: 0.7
      - config: config/moss_image_template-prodigy__feliciaday.yaml
        strength: 0.7
    output:
      path: ./output/moss_image_feliciaday_combined
      name: moss_image_feliciaday_v1_soup
      dtype: float32                           # optional; float32 keeps the merge exact

Relative paths are from the current directory, as for training. Members train one after another;
if one fails, the run stops there and nothing is merged (the LoRAs already trained stay where their
configs put them). Each member's LoRA is the ``<output.path>/<output.name>.safetensors`` its config
names, and it must have been written by this run. The merged file's path is printed last, on a line
starting ``Merged LoRA:``. Other keys in the config are ignored.

Shrinking the soup to a lower rank is a separate step: ``scripts/resize_lora.py``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import yaml
from safetensors.torch import save_file

from merge_loras import DTYPES, merge, merged_metadata

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle.read().replace("\xa0", " ")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a YAML mapping")
    return data


def load_plan(path: Path) -> Tuple[Path, List[Tuple[Path, float]], Path, str]:
    """(trainer script, [(member config, strength)], merged output file, dtype name)."""
    config = load_yaml(path)
    trainer = config.get("trainer")
    if not isinstance(trainer, str) or not trainer:
        raise ValueError(f"{path}: needs 'trainer:' naming the trainer script")
    members = config.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError(f"{path}: needs a 'members:' list of at least two configs")
    plan = []
    for i, entry in enumerate(members):
        if not isinstance(entry, dict) or not entry.get("config"):
            raise ValueError(f"{path}: members[{i}] needs a 'config'")
        strength = float(entry.get("strength", 1.0))
        if strength <= 0:
            raise ValueError(f"{path}: members[{i}] strength must be above 0, got {strength}")
        plan.append((Path(entry["config"]), strength))
    output = config.get("output") or {}
    if not output.get("path") or not output.get("name"):
        raise ValueError(f"{path}: needs 'output:' with 'path' and 'name'")
    dtype = output.get("dtype", "float32")
    if dtype not in DTYPES:
        raise ValueError(f"{path}: output dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    return Path(trainer), plan, member_output(output), dtype


def member_output(output: dict) -> Path:
    """The file a trainer writes for an ``output:`` block: ``<path>/<name>.safetensors``."""
    name = str(output.get("name") or "")
    if name.endswith(".safetensors"):
        name = name[: -len(".safetensors")]
    return Path(str(output.get("path") or "")) / f"{name}.safetensors"


def trained_lora(config_path: Path) -> Path:
    output = load_yaml(config_path).get("output") or {}
    if not output.get("path") or not output.get("name"):
        raise ValueError(f"{config_path}: has no output.path / output.name, so its LoRA cannot be found")
    return member_output(output)


def train_member(trainer: Path, config_path: Path, extra: List[str]) -> Path:
    """Run the trainer on one member config; the LoRA it wrote, or SystemExit on failure."""
    lora = trained_lora(config_path)
    started = time.time()
    command = [sys.executable, str(trainer), "--config", str(config_path), *extra]
    print(f"[combined] running: {' '.join(command)}", flush=True)
    code = subprocess.call(command)
    if code != 0:
        raise SystemExit(f"[combined] {config_path.name} failed (exit code {code}); nothing merged")
    if not lora.is_file() or lora.stat().st_mtime < started - 1:
        raise SystemExit(f"[combined] {config_path.name} finished but wrote no new {lora}; nothing merged")
    return lora


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="the combined-run YAML")
    parser.add_argument("--force", action="store_true", help="overwrite an existing merged file")
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="anything after -- is passed to every trainer run (e.g. -- --dry-run)",
    )
    args = parser.parse_args()

    try:
        trainer, members, output, dtype_name = load_plan(args.config)
        for config_path, _ in members:
            if not config_path.is_file():
                parser.error(f"member config not found: {config_path}")
            trained_lora(config_path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if not trainer.is_file() and (ROOT / trainer).is_file():
        trainer = ROOT / trainer
    if not trainer.is_file():
        parser.error(f"trainer script not found: {trainer}")
    if output.exists() and not args.force:
        parser.error(f"{output} exists; pass --force to overwrite")
    extra = [arg for arg in args.trainer_args if arg != "--"]

    trained = []
    for index, (config_path, strength) in enumerate(members, start=1):
        print(f"[combined] member {index}/{len(members)}: {config_path.name} (strength {strength})", flush=True)
        trained.append((train_member(trainer, config_path, extra), strength))
        print(f"[combined] member {index}/{len(members)} done: {trained[-1][0]}", flush=True)

    print(f"[combined] merging {len(trained)} LoRAs ({dtype_name})", flush=True)
    dtype = DTYPES[dtype_name]
    try:
        state, info, loras = merge(trained, dtype, verify=False)
    except ValueError as error:
        print(f"[combined] merge failed: {error}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output), metadata=merged_metadata(loras, info, dtype, args.config))
    print(f"[combined] {info['modules']} modules at rank {info['ranks']}", flush=True)
    print(f"Merged LoRA: {output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
