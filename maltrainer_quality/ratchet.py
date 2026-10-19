"""A limit that today's code may exceed only where a baseline says so, and only by less over time.

The baseline maps each offender to the value it had when it was recorded. The gate fails when
something not in the baseline breaks the limit, when an offender grows past its entry, or when
an entry is loose: the offender shrank, came within the limit, or is gone. ``--tighten`` lowers
loose entries and drops the ones that are within the limit. Nothing raises an entry or adds one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Verdict:
    new: dict[str, int] = field(default_factory=dict)
    grew: dict[str, tuple[int, int]] = field(default_factory=dict)
    loose: dict[str, tuple[int, int | None]] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return bool(self.new or self.grew or self.loose)


def judge(over: Mapping[str, int], baseline: Mapping[str, int]) -> Verdict:
    """Compare the offenders measured now (``over``: only what exceeds the limit) with the baseline."""
    verdict = Verdict()
    for key, value in over.items():
        allowed = baseline.get(key)
        if allowed is None:
            verdict.new[key] = value
        elif value > allowed:
            verdict.grew[key] = (value, allowed)
        elif value < allowed:
            verdict.loose[key] = (value, allowed)
    for key, allowed in baseline.items():
        if key not in over:
            verdict.loose[key] = (allowed, None)
    return verdict


def tightened(over: Mapping[str, int], baseline: Mapping[str, int]) -> dict[str, int]:
    """The baseline with every entry lowered to what was measured, and the fixed ones removed."""
    return {key: min(over[key], allowed) for key, allowed in sorted(baseline.items()) if key in over}


def load(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in entries.items()}


def save(path: Path, entries: Mapping[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(entries.items())), indent=2) + "\n", encoding="utf-8")


def report(verdict: Verdict, *, unit: str, limit: int, command: str) -> None:
    for key, value in sorted(verdict.new.items()):
        print(f"{key}: {value} {unit} (limit {limit})")
    for key, (value, allowed) in sorted(verdict.grew.items()):
        print(f"{key}: {value} {unit}, grew past its baseline of {allowed} (limit {limit})")
    for key, (allowed, value) in sorted(verdict.loose.items()):
        now = "within the limit" if value is None else f"now {value} {unit}"
        print(f"{key}: baseline {allowed} is loose, {now}")
    if verdict.loose:
        print(f"Lower the baseline with `{command} --tighten`.")


def enforce(
    over: Mapping[str, int], baseline_path: Path, *, tighten: bool, unit: str, limit: int, command: str
) -> int:
    baseline = load(baseline_path)
    if tighten:
        baseline = tightened(over, baseline)
        save(baseline_path, baseline)
    verdict = judge(over, baseline)
    report(verdict, unit=unit, limit=limit, command=command)
    return 1 if verdict.failed else 0
