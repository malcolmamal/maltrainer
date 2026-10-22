"""The package import table of docs/architecture.md (6.1), enforced: a new dependency between packages is a decision."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "maltrainer"

# Package -> packages it may import from. The three back-edges (models -> finetune, models -> training,
# utils -> finetune) are the documented exceptions; everything else points down.
ALLOWED = {
    "data": set(),
    "optimizers": set(),
    "control": {"utils"},
    "training": {"control", "data", "optimizers", "utils"},
    "utils": {"data", "finetune"},
    "finetune": {"data", "models", "utils"},
    "models": {"finetune", "training"},
    "trainers": {"control", "data", "finetune", "models", "optimizers", "training", "utils"},
}


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _modules() -> dict[str, Path]:
    return {_module_name(path): path for path in ROOT.rglob("*.py")}


def _imports(path: Path, known: set[str]) -> set[str]:
    """The maltrainer modules a file imports, each resolved to the longest module that exists."""
    targets: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        for name in names:
            parts = name.split(".")
            while parts and ".".join(parts) not in known:
                parts.pop()
            if len(parts) > 1 or (parts and parts[0] == "maltrainer" and name == "maltrainer"):
                targets.add(".".join(parts))
    return targets


def _graph() -> dict[str, set[str]]:
    modules = _modules()
    known = set(modules)
    return {name: _imports(path, known) - {name} for name, path in modules.items()}


def test_packages_import_only_from_the_packages_the_architecture_table_allows() -> None:
    found: dict[str, set[str]] = defaultdict(set)
    for module, targets in _graph().items():
        source = module.split(".")[1] if module.count(".") >= 2 else None
        for target in targets:
            destination = target.split(".")[1] if target.count(".") >= 1 else None
            if source and destination and source != destination:
                found[source].add(destination)

    unexpected = {
        pkg: sorted(found[pkg] - ALLOWED.get(pkg, set())) for pkg in found if found[pkg] - ALLOWED.get(pkg, set())
    }
    assert unexpected == {}, "update ALLOWED and docs/architecture.md 6.1 together if this is intended"


def test_nothing_in_the_library_imports_the_trainers_or_an_entry_point() -> None:
    for module, targets in _graph().items():
        if module.split(".")[1:2] != ["trainers"]:
            assert not any(t.startswith("maltrainer.trainers") for t in targets), module
        assert not any(t.startswith("train_") for t in targets), module


def test_the_module_import_graph_has_no_cycle() -> None:
    graph = _graph()
    state: dict[str, int] = {}

    def visit(module: str, trail: list[str]) -> None:
        state[module] = 1
        for target in sorted(graph.get(module, ())):
            if state.get(target) == 1:
                raise AssertionError("import cycle: " + " -> ".join([*trail[trail.index(target) :], module, target]))
            if target not in state:
                visit(target, [*trail, module] if module not in trail else trail)
        state[module] = 2

    for module in sorted(graph):
        if module not in state:
            visit(module, [])
