"""Mutation targets: the optimizers (see ``mutation_targets`` for what a target is)."""

from __future__ import annotations

from maltrainer_quality.mutation_target import Target

AUTOMAGIC3_TESTS = (
    "tests/maltrainer/optimizers/test_automagic3_bits.py",
    "tests/maltrainer/optimizers/test_automagic3_groups.py",
    "tests/maltrainer/optimizers/test_automagic3_lr.py",
    "tests/maltrainer/optimizers/test_automagic3_precision.py",
    "tests/maltrainer/optimizers/test_automagic3_state.py",
    "tests/maltrainer/optimizers/test_automagic3_step.py",
    "tests/maltrainer/optimizers/test_automagic3_update_math.py",
)
RESUME_TESTS = ("tests/maltrainer/optimizers/test_automagic3_resume_history.py",)

OPTIMIZERS_TARGETS: tuple[Target, ...] = (
    Target(
        "maltrainer/optimizers/automagic3_ops.py",
        AUTOMAGIC3_TESTS,
        94.0,
    ),
    Target(
        "maltrainer/optimizers/automagic3_update.py",
        AUTOMAGIC3_TESTS,
        97.0,
    ),
    Target(
        "maltrainer/optimizers/automagic3_stats.py",
        AUTOMAGIC3_TESTS,
        95.0,
    ),
    Target(
        "maltrainer/optimizers/automagic3_resume.py",
        AUTOMAGIC3_TESTS + RESUME_TESTS,
        93.0,
    ),
    Target(
        "maltrainer/optimizers/automagic3_base.py",
        AUTOMAGIC3_TESTS + RESUME_TESTS,
        98.0,
    ),
)
