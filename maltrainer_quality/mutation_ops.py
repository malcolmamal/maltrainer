"""The mutation operators: every way this runner changes one expression of a module.

A *site* is one mutation: ``sites(source)`` lists them in a fixed order and ``mutate(source, index)``
returns the module's code with only that site applied. Each operator is the small, classic kind that a
test with a pinned value or a boundary case notices: an arithmetic or comparison operator swapped
for its neighbour, ``and`` for ``or``, a constant moved by one, a sign or a ``not`` dropped, a return
value replaced by ``None``. Annotations, docstrings, ``raise`` statements (their messages) and
``TYPE_CHECKING`` blocks are never mutated: they change no behaviour a test could pin.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from types import CodeType
from typing import cast

BINARY: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Mult,
    ast.Pow: ast.Mult,
    ast.BitAnd: ast.BitOr,
    ast.BitOr: ast.BitAnd,
    ast.BitXor: ast.BitAnd,
    ast.LShift: ast.RShift,
    ast.RShift: ast.LShift,
}
COMPARE: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


@dataclass(frozen=True)
class Site:
    """One mutation: its position in ``sites(source)``, where it is, and what it changes."""

    index: int
    line: int
    label: str


def _show(node: ast.AST) -> str:
    text = ast.unparse(node).replace("\n", " ")
    return text if len(text) <= 48 else text[:45] + "..."


def _constant(node: ast.Constant) -> list[ast.AST]:
    value = node.value
    if isinstance(value, bool):
        return [ast.Constant(value=not value)]
    if isinstance(value, (int, float)):
        return [ast.Constant(value=value + 1)]
    if isinstance(value, str) and value:
        return [ast.Constant(value=f"XX{value}XX")]
    return []


def _compare(node: ast.Compare) -> list[ast.AST]:
    out: list[ast.AST] = []
    for position, operator in enumerate(node.ops):
        changed = copy.copy(node)
        changed.ops = [*node.ops]
        changed.ops[position] = COMPARE[type(operator)]()
        out.append(changed)
    return out


def _return(node: ast.Return) -> list[ast.AST]:
    value = node.value
    if value is None or (isinstance(value, ast.Constant) and value.value is None):
        return []
    return [ast.Return(value=ast.Constant(value=None))]


def _unary(node: ast.UnaryOp) -> list[ast.AST]:
    if isinstance(node.op, (ast.Not, ast.USub)):
        return [node.operand]
    return []


def _binary(node: ast.BinOp | ast.AugAssign) -> list[ast.AST]:
    swapped = BINARY.get(type(node.op))
    if swapped is None:
        return []
    changed = copy.copy(node)
    changed.op = swapped()
    return [changed]


def _boolean(node: ast.BoolOp) -> list[ast.AST]:
    changed = copy.copy(node)
    changed.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    return [changed]


_DEFS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
_GUARDS = {"TYPE_CHECKING", "typing.TYPE_CHECKING", "__name__ == '__main__'"}


def _docstring(node: ast.AST) -> list[ast.AST]:
    body = getattr(node, "body", [])
    first = body[0] if body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        return [first.value]
    return []


def _annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    found: list[ast.AST] = [a.annotation for a in ast.walk(node.args) if isinstance(a, ast.arg) and a.annotation]
    return [*found, node.returns] if node.returns else found


def _shielded(node: ast.AST) -> list[ast.AST]:
    """The nodes below which nothing is mutated, as decided by ``node`` itself."""
    found = _docstring(node) if isinstance(node, _DEFS) else []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        found += _annotations(node)
    if isinstance(node, ast.AnnAssign):
        found.append(node.annotation)
    if isinstance(node, (ast.Raise, ast.JoinedStr)):
        found.append(node)
    if isinstance(node, ast.If) and ast.unparse(node.test) in _GUARDS:
        found += [node.test, *node.body]
    if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "__all__":
        found.append(node)
    return found


def _protected(tree: ast.Module) -> set[int]:
    """Node ids that are never mutated, with everything below them."""
    return {id(shielded) for node in ast.walk(tree) for shielded in _shielded(node)}


class _Mutator(ast.NodeTransformer):
    def __init__(self, skip: set[int], target: int | None) -> None:
        self.skip = skip
        self.target = target
        self.sites: list[Site] = []

    def _offer(self, node: ast.AST, alternatives: list[ast.AST]) -> ast.AST:
        chosen = node
        for alternative in alternatives:
            index = len(self.sites)
            line = getattr(node, "lineno", 0)
            self.sites.append(Site(index, line, f"{_show(node)}  ->  {_show(alternative)}"))
            if index == self.target:
                chosen = ast.copy_location(alternative, node)
        return chosen

    def visit(self, node: ast.AST) -> ast.AST:
        if id(node) in self.skip:
            return node
        node = self.generic_visit(node)
        if isinstance(node, ast.Constant):
            return self._offer(node, _constant(node))
        if isinstance(node, ast.Compare):
            return self._offer(node, _compare(node))
        if isinstance(node, ast.Return):
            return self._offer(node, _return(node))
        if isinstance(node, ast.UnaryOp):
            return self._offer(node, _unary(node))
        if isinstance(node, (ast.BinOp, ast.AugAssign)):
            return self._offer(node, _binary(node))
        if isinstance(node, ast.BoolOp):
            return self._offer(node, _boolean(node))
        return node


def _run(source: str, target: int | None) -> tuple[ast.Module, list[Site]]:
    tree = ast.parse(source)
    mutator = _Mutator(_protected(tree), target)
    changed = cast(ast.Module, mutator.visit(tree))
    return ast.fix_missing_locations(changed), mutator.sites


def sites(source: str) -> list[Site]:
    """Every mutation ``mutate`` can apply to ``source``, in a stable order."""
    return _run(source, None)[1]


def mutate(source: str, index: int, filename: str = "<mutant>") -> CodeType:
    """The compiled module with only mutation number ``index`` applied."""
    return compile(_run(source, index)[0], filename, "exec")
