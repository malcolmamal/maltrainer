"""The pure-math modules under mutation testing, the tests that cover each, and its score floor.

``tests`` are the files a mutant of that module is run against (``-x``, so the first failure kills it): the
unit tests that pin the module's maths, not the trainer end-to-end tests. ``floor`` is the lowest mutation
score (percent killed) the module may have; it sits a little below the measured value and is only raised
(docs/mutation-testing.md). The rows live in ``mutation_targets_logic`` (training, planning, finetune),
``mutation_targets_models`` and ``mutation_targets_optimizers``; add a module by adding a row there
(``test_mutation_run`` checks the files exist).
"""

from __future__ import annotations

from maltrainer_quality.mutation_target import Target
from maltrainer_quality.mutation_targets_logic import LOGIC_TARGETS
from maltrainer_quality.mutation_targets_models import MODELS_TARGETS
from maltrainer_quality.mutation_targets_optimizers import OPTIMIZERS_TARGETS

TARGETS: tuple[Target, ...] = LOGIC_TARGETS + MODELS_TARGETS + OPTIMIZERS_TARGETS

__all__ = ["TARGETS", "Target"]
