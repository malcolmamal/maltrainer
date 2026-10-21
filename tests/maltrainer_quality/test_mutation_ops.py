from __future__ import annotations

from typing import Any

import pytest

from maltrainer_quality.mutation_ops import mutate, sites


def labels(source: str) -> list[str]:
    return [site.label for site in sites(source)]


def run(source: str, index: int) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(mutate(source, index), namespace)  # noqa: S102
    return namespace


@pytest.mark.parametrize(
    ("expression", "changed"),
    [
        ("a + b", "a - b"),
        ("a - b", "a + b"),
        ("a * b", "a / b"),
        ("a < b", "a <= b"),
        ("a >= b", "a > b"),
        ("a == b", "a != b"),
        ("a is None", "a is not None"),
        ("a and b", "a or b"),
        ("not a", "a"),
        ("-a", "a"),
    ],
)
def test_an_operator_is_swapped_for_its_neighbour(expression: str, changed: str) -> None:
    assert f"{expression}  ->  {changed}" in labels(f"def f(a, b):\n    x = {expression}\n")


def test_constants_move_by_one_and_booleans_flip() -> None:
    found = labels("x = 3\ny = 0.5\nz = True\nw = 'k'\nv = ''\nu = None\n")

    assert found == ["3  ->  4", "0.5  ->  1.5", "True  ->  False", "'k'  ->  'XXkXX'"]


def test_a_return_value_becomes_none_unless_it_is_none() -> None:
    assert labels("def f(a):\n    return a\n") == ["return a  ->  return None"]
    assert labels("def f():\n    return None\n    return\n") == []


def test_nothing_is_mutated_that_no_test_could_pin() -> None:
    source = '''"""Doc 1."""
from __future__ import annotations
from typing import TYPE_CHECKING, Literal
__all__ = ["f"]
if TYPE_CHECKING:
    X = 1 + 2
def f(a: Literal["k"] = None, *rest: int | None, b: int | None = None) -> int | None:
    """Doc 2."""
    value: Literal["q"] = a
    if a:
        raise ValueError(f"bad {a + 1}" + "text")
    return None
class C:
    """Doc 3."""
if __name__ == "__main__":
    f(5)
'''
    assert labels(source) == []


def test_a_comparison_chain_is_mutated_one_operator_at_a_time() -> None:
    assert labels("x = a < b <= c\n") == ["a < b <= c  ->  a <= b <= c", "a < b <= c  ->  a < b < c"]


def test_only_the_chosen_site_is_applied() -> None:
    source = "def f(a, b):\n    return a + b * 2\n"
    found = labels(source)
    assert len(found) == 4
    results = [run(source, index)["f"](3, 5) for index in range(len(found))]

    assert results == [3 + 5 * 3, 3 + 5 / 2, 3 - 5 * 2, None]


def test_a_site_reports_its_line_and_the_mutant_keeps_the_filename() -> None:
    found = sites("x = 1\ny = 2\n")
    assert [(site.index, site.line) for site in found] == [(0, 1), (1, 2)]
    assert mutate("x = 1\n", 0, "m.py").co_filename == "m.py"


def test_long_expressions_are_shortened_in_labels() -> None:
    long = " + ".join(["aaaaaaaa"] * 8)

    assert all(len(label) <= 2 * 48 + 6 for label in labels(f"x = {long}\n"))
    assert any("..." in label for label in labels(f"x = {long}\n"))


def test_operators_without_a_neighbour_are_left_alone() -> None:
    assert labels("def f(a, b):\n    return ~a @ b\n") == ["return ~a @ b  ->  return None"]
