"""Mutation targets: the training, planning and finetune logic (see ``mutation_targets`` for what a target is)."""

from __future__ import annotations

from maltrainer_quality.mutation_target import Target

LOGIC_TARGETS: tuple[Target, ...] = (
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
        "maltrainer/utils/ema.py",
        (
            "tests/maltrainer/utils/test_ema.py",
            "tests/maltrainer/utils/test_ema_defaults.py",
        ),
        74.0,
    ),
)
