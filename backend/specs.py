"""What the UI and the planner know about each trainable model family."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    label: str
    default_lr: float
    note: str = ""


@dataclass(frozen=True)
class ModelSpec:
    """One trainable model family: which script runs it, which template seeds its config."""

    id: str
    label: str
    script: str
    template: str
    optimizers: tuple[OptimizerSpec, ...]
    default_optimizer: str
    default_duration_mode: str  # "epochs" | "steps"
    default_epochs: int
    default_max_steps: int
    # True when the trainer honours train.max_steps itself. Otherwise a step budget is
    # rounded up to whole epochs before the config is written.
    native_max_steps: bool
    # Required checkpoint files, found anywhere under COMFYUI_MODELS_PATH/<subfolder>/.
    model_files: tuple[tuple[str, str, str], ...] = ()  # (config key, subfolder, file name)
    # Measured seconds per step (RTX 5090, 512px, rank 16) for the UI's time estimate; 0 = unknown.
    sec_per_step: float = 0.0
    setup_seconds: float = 0.0
    # Legacy runs write straight to output/<dataset>; newer models get one folder per run.
    per_run_output: bool = False
    # Auto step budget, used when the user gives no step count: aim for target_epochs epochs of
    # this dataset, rounded to 50 and kept inside [min_auto_steps, max_auto_steps] -- the range
    # the benchmark actually covered. target_epochs 0 = no auto budget.
    target_epochs: int = 0
    min_auto_steps: int = 0
    max_auto_steps: int = 0
    # Guidance derived from the benchmark (0 / False = off).
    epoch_bounds: tuple[int, int] = (0, 0)  # warn below the first / above the second
    max_benchmarked_steps: int = 0
    # Training is square-only for now: non-square images are squashed, so the plan warns about them.
    square_only: bool = False
    resolution_note: str = ""
    # A whole recipe per optimizer, where the optimizer changes more than its learning rate.
    optimizer_templates: tuple[tuple[str, str], ...] = ()  # (optimizer, template path)
    # Output file prefix (the name the LoRA pickers parse) and whether a custom trigger also
    # rewrites the default caption to "a photo of <trigger>".
    output_prefix: str = ""
    caption_from_trigger: bool = False
