"""The child side of the mutation runner: a process that runs one mutant at a time.

Importing torch takes seconds, so a fresh pytest process per mutant would spend nearly all its time
there. The worker imports it once, and for every job drops the project's modules from ``sys.modules``,
puts a finder in front of the target module that serves the mutated code, and runs pytest in this
process. The unmutated source is never written anywhere: no temporary tree, nothing for a concurrent
edit of the checkout to trip over.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import sys
import time
from collections.abc import MutableMapping, Sequence
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder, SourceFileLoader
from multiprocessing.connection import Connection
from types import CodeType
from typing import Any

import pytest

from maltrainer_quality.mutation_ops import mutate

# Everything the tests import from the checkout; third-party packages stay loaded.
PROJECT_ROOTS = {"maltrainer", "maltrainer_quality", "backend", "scripts", "tests", "conftest"}
PYTEST_ARGUMENTS = ("-x", "-q", "--no-cov", "-p", "no:cacheprovider")
READY = "ready"


class _MutantLoader(SourceFileLoader):
    def __init__(self, fullname: str, path: str, code: CodeType) -> None:
        super().__init__(fullname, path)
        self._code = code

    def get_code(self, _fullname: str) -> CodeType:
        return self._code


class _MutantFinder(MetaPathFinder):
    def __init__(self, module: str, code: CodeType) -> None:
        self.module = module
        self.code = code

    def find_spec(self, fullname: str, path: Sequence[str] | None, _target: object = None) -> ModuleSpec | None:
        if fullname != self.module:
            return None
        found = PathFinder.find_spec(fullname, list(path) if path is not None else None)
        if found is None or found.origin is None:
            return None
        loader = _MutantLoader(fullname, found.origin, self.code)
        return importlib.util.spec_from_file_location(fullname, found.origin, loader=loader)


def forget_project(module: str, loaded: MutableMapping[str, object] | None = None) -> None:
    """Drop the project's modules (and the target's package) from ``loaded`` (``sys.modules``)."""
    table: MutableMapping[str, Any] = sys.modules if loaded is None else loaded
    roots = {*PROJECT_ROOTS, module.partition(".")[0]}
    for name in [n for n in table if n.partition(".")[0] in roots or n.startswith("train_")]:
        del table[name]


def run_job(module: str, source: str, index: int | None, tests: Sequence[str]) -> tuple[int, float, str]:
    """Run ``tests`` against ``module`` with mutation ``index`` applied (none: unmutated).

    Returns pytest's exit code, the seconds it took and the tail of its output.
    """
    forget_project(module)
    finder = None if index is None else _MutantFinder(module, mutate(source, index, module))
    if finder is not None:
        sys.meta_path.insert(0, finder)
    out = io.StringIO()
    started = time.monotonic()
    try:
        with contextlib.redirect_stdout(out):
            code = int(pytest.main([*tests, *PYTEST_ARGUMENTS]))
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)
    return code, time.monotonic() - started, out.getvalue()[-2000:]


def serve(connection: Connection, preload: Sequence[str]) -> None:
    """Entry point of the worker process: preload, say ready, then run jobs until told to stop."""
    for name in preload:
        importlib.import_module(name)
    connection.send(READY)
    while (job := connection.recv()) is not None:
        connection.send(run_job(*job))
