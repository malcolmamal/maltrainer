"""What the benchmark says about a planned run: the step budget to pick for a dataset, and warnings for runs too
short, too long or longer than anything measured."""

from __future__ import annotations

from backend.specs import ModelSpec


def steps_for_dataset(spec: ModelSpec, image_count: int, steps_per_epoch: int) -> tuple[int, str]:
    """The step budget for a dataset when the user gave none, and the note that says how it was chosen."""
    wanted = spec.target_epochs * steps_per_epoch
    rounded = round(wanted / 50.0) * 50
    max_steps = min(max(rounded, spec.min_auto_steps), spec.max_auto_steps)
    note = f"Steps chosen automatically: {image_count} images × ~{spec.target_epochs} epochs = {wanted} steps"
    if rounded != wanted:
        note += f", rounded to {rounded}"
    if max_steps != rounded:
        note += (
            f", then kept within {spec.min_auto_steps}–{spec.max_auto_steps} steps, the range the "
            f"benchmark covered, giving {max_steps}"
        )
    return max_steps, note + "."


def run_length_warnings(spec: ModelSpec, effective_epochs: float | None, total_steps: int, warnings: list[str]) -> None:
    """Warnings from the benchmark: too few or too many epochs, a longer run than any measured."""
    if effective_epochs is not None and spec.epoch_bounds != (0, 0):
        low, high = spec.epoch_bounds
        if effective_epochs < low:
            warnings.append(
                f"Only {effective_epochs:.1f} epochs. In the benchmark, likeness was at roughly 70% of its "
                f"plateau after 11 epochs and 90–95% after 22, so runs under ~{low} epochs risk being "
                "under-trained. Raise max steps if you can."
            )
        elif effective_epochs > high:
            warnings.append(
                f"{effective_epochs:.1f} epochs is beyond what was benchmarked (up to {high - 1}); "
                "very small datasets can over-fit or memorize."
            )
    if spec.max_benchmarked_steps and total_steps > spec.max_benchmarked_steps:
        warnings.append(
            f"{total_steps} steps is longer than any benchmarked run ({spec.max_benchmarked_steps} at most). "
            "Automagic3 stayed stable up to there; beyond it is untested."
        )
