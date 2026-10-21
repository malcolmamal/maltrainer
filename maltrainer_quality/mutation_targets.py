"""The pure-math modules under mutation testing, the tests that cover each, and its score floor.

``tests`` are the files a mutant of that module is run against (``-x``, so the first failure kills it): the
unit tests that pin the module's maths, not the trainer end-to-end tests. ``floor`` is the lowest mutation
score (percent killed) the module may have; it sits a little below the measured value and is only raised
(docs/mutation-testing.md). Add a module by adding a row; ``test_mutation_run`` checks the files exist.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    path: str
    tests: tuple[str, ...]
    floor: float


TARGETS: tuple[Target, ...] = (
    Target(
        "maltrainer/training/steps.py",
        (
            "tests/maltrainer/training/test_base.py",
            "tests/maltrainer/training/test_steps.py",
            "tests/trainers/minimax_h3/test_epoch_order.py",
        ),
        94.0,
    ),
    Target(
        "maltrainer/training/sigma.py",
        (
            "tests/maltrainer/training/test_sigma.py",
            "tests/trainers/flux3_klein/test_sigma.py",
            "tests/trainers/krea2/test_lora_sigma.py",
            "tests/trainers/minimax_h3/test_sigma.py",
            "tests/trainers/moss_image/test_lora_sigma.py",
            "tests/trainers/qwen_image_2_1/test_sigma.py",
        ),
        98.0,
    ),
    Target(
        "maltrainer/models/scheduler.py",
        ("tests/maltrainer/models/test_scheduler.py",),
        98.0,
    ),
    Target(
        "maltrainer/models/krea2/timesteps.py",
        ("tests/trainers/krea2/test_lora_sigma.py",),
        98.0,
    ),
    Target(
        "maltrainer/models/moss_image/model_math.py",
        (
            "tests/maltrainer/models/moss_image/test_model_math.py",
            "tests/maltrainer/models/moss_image/test_model_forward.py",
        ),
        96.0,
    ),
    Target(
        "maltrainer/models/moss_image/sampling.py",
        ("tests/maltrainer/models/moss_image/test_sampling.py",),
        92.0,
    ),
    Target(
        "maltrainer/models/qwen_image_2_1/model_math.py",
        (
            "tests/maltrainer/models/qwen_image_2_1/test_model_rope.py",
            "tests/maltrainer/models/qwen_image_2_1/test_model_schedule.py",
            "tests/maltrainer/models/qwen_image_2_1/test_model_transformer.py",
        ),
        91.0,
    ),
    Target(
        "maltrainer/models/flux3/rope.py",
        (
            "tests/maltrainer/models/flux3/test_model_math.py",
            "tests/maltrainer/models/flux3/test_model_blocks.py",
        ),
        93.0,
    ),
    Target(
        "maltrainer/models/krea2/rope.py",
        (
            "tests/maltrainer/models/krea2/test_model_math.py",
            "tests/maltrainer/models/krea2/test_model_blocks.py",
        ),
        96.0,
    ),
    Target(
        "maltrainer/optimizers/automagic3_ops.py",
        (
            "tests/maltrainer/optimizers/test_automagic3_bits.py",
            "tests/maltrainer/optimizers/test_automagic3_groups.py",
            "tests/maltrainer/optimizers/test_automagic3_lr.py",
            "tests/maltrainer/optimizers/test_automagic3_precision.py",
            "tests/maltrainer/optimizers/test_automagic3_state.py",
            "tests/maltrainer/optimizers/test_automagic3_step.py",
            "tests/maltrainer/optimizers/test_automagic3_update_math.py",
        ),
        94.0,
    ),
    Target(
        "maltrainer/optimizers/automagic3_update.py",
        (
            "tests/maltrainer/optimizers/test_automagic3_bits.py",
            "tests/maltrainer/optimizers/test_automagic3_groups.py",
            "tests/maltrainer/optimizers/test_automagic3_lr.py",
            "tests/maltrainer/optimizers/test_automagic3_precision.py",
            "tests/maltrainer/optimizers/test_automagic3_state.py",
            "tests/maltrainer/optimizers/test_automagic3_step.py",
            "tests/maltrainer/optimizers/test_automagic3_update_math.py",
        ),
        97.0,
    ),
    Target(
        "maltrainer/optimizers/automagic3_stats.py",
        (
            "tests/maltrainer/optimizers/test_automagic3_bits.py",
            "tests/maltrainer/optimizers/test_automagic3_groups.py",
            "tests/maltrainer/optimizers/test_automagic3_lr.py",
            "tests/maltrainer/optimizers/test_automagic3_precision.py",
            "tests/maltrainer/optimizers/test_automagic3_state.py",
            "tests/maltrainer/optimizers/test_automagic3_step.py",
            "tests/maltrainer/optimizers/test_automagic3_update_math.py",
        ),
        95.0,
    ),
    Target(
        "maltrainer/utils/step_planner.py",
        ("tests/maltrainer/utils/test_step_planner.py",),
        95.0,
    ),
    Target(
        "maltrainer/finetune/repeats.py",
        ("tests/maltrainer/finetune/test_repeats.py",),
        98.0,
    ),
    Target(
        "maltrainer/finetune/selection.py",
        ("tests/maltrainer/finetune/test_selection.py",),
        98.0,
    ),
    Target(
        "maltrainer/finetune/classes.py",
        (
            "tests/maltrainer/finetune/test_classes.py",
            "tests/maltrainer/finetune/test_classes_one_pass.py",
        ),
        91.0,
    ),
    Target(
        "maltrainer/models/minimax_h3/int8_rotation.py",
        (
            "tests/maltrainer/models/minimax_h3/test_int8_convrot_rotation.py",
            "tests/maltrainer/models/minimax_h3/test_int8_convrot_function.py",
            "tests/maltrainer/models/minimax_h3/test_int8_convrot_dispatch.py",
        ),
        88.0,
    ),
)
