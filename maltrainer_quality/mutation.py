"""Mutation testing for the pure-math modules: ``python -m maltrainer_quality.mutation``.

Not part of the default gate (it takes minutes); see docs/mutation-testing.md for when it must be run.
Each mutant (one changed expression, see ``mutation_ops``) is tested by only the test files that
cover its module (``mutation_targets``). A mutant the tests notice is killed; one they miss survived.
The score is killed (timeouts included) over all mutants, and a module fails when it is below its floor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from maltrainer_quality.mutation_ops import Site, sites
from maltrainer_quality.mutation_pool import Job, Pool
from maltrainer_quality.mutation_select import changed_files, select_changed
from maltrainer_quality.mutation_targets import TARGETS, Target

MIN_TIMEOUT = 30.0
TIMEOUT_FACTOR = 6.0
NO_TESTS_COLLECTED = 5


@dataclass(frozen=True)
class Report:
    target: Target
    total: int
    survivors: tuple[Site, ...]
    seconds: float

    @property
    def score(self) -> float:
        return 100.0 if self.total == 0 else 100.0 * (self.total - len(self.survivors)) / self.total

    @property
    def passed(self) -> bool:
        return self.score >= self.target.floor


def module_name(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def run_target(pool: Pool, target: Target, root: Path) -> Report:
    started = time.monotonic()
    source = (root / target.path).read_text(encoding="utf-8")
    module, tests = module_name(target.path), tuple(str(root / test) for test in target.tests)
    baseline = pool.one((module, source, None, tests), 1800.0)
    if baseline.code != 0:
        code = "no tests ran" if baseline.code == NO_TESTS_COLLECTED else f"exit code {baseline.code}"
        raise SystemExit(f"{target.path}: the unmutated tests fail ({code}):\n{baseline.output}")
    found = sites(source)
    jobs: list[Job] = [(module, source, site.index, tests) for site in found]
    outcomes = pool.map(jobs, max(MIN_TIMEOUT, TIMEOUT_FACTOR * baseline.seconds))
    survivors = tuple(site for site, outcome in zip(found, outcomes, strict=True) if outcome.survived)
    return Report(target, len(found), survivors, time.monotonic() - started)


def describe(report: Report) -> str:
    target = report.target
    killed = report.total - len(report.survivors)
    mark = "ok" if report.passed else "BELOW FLOOR"
    lines = [f"{target.path}: {killed}/{report.total} killed, {report.score:.1f}% (floor {target.floor}) {mark}"]
    lines += [f"    survived {target.path}:{site.line}: {site.label}" for site in report.survivors]
    return "\n".join(lines)


def to_json(reports: Sequence[Report]) -> str:
    entries = {
        report.target.path: {
            "total": report.total,
            "survived": [{"line": s.line, "change": s.label} for s in report.survivors],
            "score": round(report.score, 2),
            "floor": report.target.floor,
            "seconds": round(report.seconds, 1),
        }
        for report in reports
    }
    return json.dumps(entries, indent=2)


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(prog="maltrainer_quality.mutation", description=__doc__)
    cli.add_argument("modules", nargs="*", help="target module paths, e.g. maltrainer/training/sigma.py")
    cli.add_argument("--all", action="store_true", help="every target module")
    cli.add_argument("--changed", action="store_true", help="the targets whose module or tests changed (git)")
    cli.add_argument("--base", default="origin/main", help="what --changed compares with (default origin/main)")
    cli.add_argument("--list", action="store_true", help="list the targets, their tests and floors, and stop")
    cli.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="worker processes")
    cli.add_argument("--json", type=Path, help="also write the result as JSON here")
    return cli


def chosen(arguments: argparse.Namespace, root: Path) -> list[Target]:
    if arguments.all:
        return list(TARGETS)
    if arguments.changed:
        return select_changed(changed_files(root, arguments.base))
    by_path = {target.path: target for target in TARGETS}
    unknown = [path for path in arguments.modules if path not in by_path]
    if unknown or not arguments.modules:
        raise SystemExit(f"name target modules, or use --all / --changed. Unknown: {unknown}")
    return [by_path[path] for path in arguments.modules]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    root = Path.cwd()
    if arguments.list:
        print("\n".join(f"{t.path}  floor {t.floor}  tests: {' '.join(t.tests)}" for t in TARGETS))
        return 0
    targets = chosen(arguments, root)
    if not targets:
        print("no mutation target module or its tests changed")
        return 0
    reports: list[Report] = []
    with Pool(arguments.jobs) as pool:
        for target in targets:
            reports.append(run_target(pool, target, root))
            print(describe(reports[-1]), flush=True)
    if arguments.json is not None:
        arguments.json.write_text(to_json(reports), encoding="utf-8")
    print(f"{len(reports)} module(s) in {sum(r.seconds for r in reports):.0f} s of worker time")
    return 0 if all(report.passed for report in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
