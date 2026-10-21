# Mutation testing

Line coverage says a test *ran* a line, not that it would notice the line being wrong. Mutation testing
changes the code one expression at a time (a `+` into a `-`, a `<` into a `<=`, a constant moved by one, a
return value dropped) and runs the tests. A failing run has **killed** the mutant; one that still passes means
the tests do not pin that behaviour: the mutant **survived**. The score is killed over all mutants. It is run
on the pure-math modules only (RoPE, sigma and schedule maths, optimizer maths, step planning, the int8
rotation): fast CPU tests, no model weights, and a wrong constant there trains a worse model without raising
anything.

It is **not** in the default gate (`python -m maltrainer_quality`): the whole set takes five to eight minutes.
When it must be run is under "Policy" below.

## Running it

```bash
.venv/Scripts/python -m maltrainer_quality.mutation --changed        # the targets a change touches (git)
.venv/Scripts/python -m maltrainer_quality.mutation --all            # every target, 5 to 8 minutes
.venv/Scripts/python -m maltrainer_quality.mutation maltrainer/training/sigma.py   # one or more modules
.venv/Scripts/python -m maltrainer_quality.mutation --list           # targets, floors and the tests of each
```

`--changed` compares with `origin/main` (`--base REF` for another branch): the commits since the branch forked,
the working tree and untracked files. A target is selected when its module **or any test file mapped to it**
changed. `--jobs N` sets the worker processes (default 4; each imports torch once, about 10 s).
`--json FILE` also writes the result. The exit code is 1 when a module is below its floor.

Output, per module:

```text
maltrainer/training/sigma.py: 24/24 killed, 100.0% (floor 98.0) ok
maltrainer/models/flux3/rope.py: 65/68 killed, 95.6% (floor 93.0) ok
    survived maltrainer/models/flux3/rope.py:22: -1  ->  1
```

Each `survived` line is the file, the line and the change (`original  ->  mutant`). For every one decide:

1. **Real gap**: the changed code would be a bug. Add an assertion that pins the value (an exact expected number
   computed independently, a boundary case, a non-default argument) in the module's own unit test, rerun the
   module and see the mutant die. One precise assertion beats a longer test.
2. **Equivalent mutant**: the changed code behaves the same for every input the product can produce (a clamp
   that never triggers, `dim=-1` against `dim=1` on a 2-D tensor, a guard for a second device). Nothing can
   kill it. Add it to the table below with the reason; it stays in the score as a survivor, which is why the
   floors are below 100.
3. **A real bug**: the test you write fails on the product code. Follow AGENTS.md (failing test first, say in
   the commit message which area it affects). Do not edit product code to make a mutant die.

## How it works

Fork-based tools do not run on Windows: mutmut 3 stops with "please use the WSL" (tried), and tools that start a
test process per mutant would spend nearly all their time importing torch (9 s). `maltrainer_quality/mutation*.py`
is therefore a small runner of our own:

- `mutation_ops.py`: the AST operators. Arithmetic, bitwise and shift operators swapped for a neighbour;
  comparisons moved across their boundary (`<` to `<=`, `==` to `!=`, `is`, `in`); `and` / `or`; `not` and unary
  minus dropped; integer and float constants plus one, booleans flipped, non-empty strings changed; `return
  value` turned into `return None`. Never mutated: annotations, docstrings, `raise` statements (their messages),
  f-strings, `TYPE_CHECKING` and `__main__` blocks, `__all__`.
- `mutation_worker.py`: a long-lived process per worker. For each mutant it forgets the project's modules, puts a
  finder in front of the target module that serves the mutated code, and runs pytest in the same process
  (`-x -q --no-cov -p no:cacheprovider`) on only the test files mapped to the module. Nothing is written to
  disk, so the checkout and `output/` are untouched and a concurrent edit of the checkout cannot disturb a run.
- `mutation_pool.py`: the workers and the pool. A mutant that makes the tests run past a time limit (6 times
  the unmutated run, at least 30 s: an endless loop) or kills the worker counts as killed; the worker is replaced.
- `mutation_targets.py` (rows in `mutation_targets_logic.py`, `_models.py` and `_optimizers.py`): the table of target modules, their tests and floors. `mutation_select.py`: `--changed`.
- Before mutating a module, its unmutated tests must pass in a worker, or the run stops with pytest's output.

The tests of a module are the unit tests that pin its maths (found with coverage contexts: which test files
execute the module), not the end-to-end trainer tests, which also cover it but cost minutes each. If a mutant
survives only because a trainer test would have caught it, the unit test is the one that is missing.

## Scores

Measured on Windows, CPU, 4 workers. Before is the first run on the tests as they were; after is with the
assertions added in the same commit as this document. The floor is the measured whole percent minus 2.

| Module | Killed before | % | Killed after | % | Floor |
|---|---|---|---|---|---|
| `training/steps.py` | 5 / 33 | 15.2 | 32 / 33 | 97.0 | 94 |
| `training/sigma.py` | 22 / 24 | 91.7 | 24 / 24 | 100.0 | 98 |
| `models/scheduler.py` | 4 / 4 | 100.0 | 4 / 4 | 100.0 | 98 |
| `models/krea2/timesteps.py` (a) | 29 / 29 | 100.0 | 29 / 29 | 100.0 | 98 |
| `models/moss_image/model_math.py` | 94 / 96 | 97.9 | 95 / 96 | 99.0 | 96 |
| `models/moss_image/sampling.py` | 61 / 71 | 85.9 | 67 / 71 | 94.4 | 92 |
| `models/qwen_image_2_1/model_math.py` | 59 / 65 | 90.8 | 61 / 65 | 93.8 | 91 |
| `models/flux3/rope.py` | 60 / 68 | 88.2 | 65 / 68 | 95.6 | 93 |
| `models/krea2/rope.py` | 48 / 50 | 96.0 | 49 / 50 | 98.0 | 96 |
| `optimizers/automagic3_ops.py` | 46 / 61 | 75.4 | 59 / 61 | 96.7 | 94 |
| `optimizers/automagic3_update.py` | 103 / 122 | 84.4 | 121 / 122 | 99.2 | 97 |
| `optimizers/automagic3_stats.py` | 31 / 36 | 86.1 | 35 / 36 | 97.2 | 95 |
| `utils/step_planner.py` | 37 / 41 | 90.2 | 40 / 41 | 97.6 | 95 |
| `finetune/repeats.py` | 21 / 21 | 100.0 | 21 / 21 | 100.0 | 98 |
| `finetune/selection.py` | 11 / 11 | 100.0 | 11 / 11 | 100.0 | 98 |
| `finetune/classes.py` | 14 / 15 | 93.3 | 14 / 15 | 93.3 | 91 |
| `models/minimax_h3/int8_rotation.py` | 66 / 77 | 85.7 | 70 / 77 | 90.9 | 88 |
| `optimizers/automagic3_resume.py` (b) | 39 / 47 | 83.0 | 45 / 47 | 95.7 | 93 |
| `optimizers/automagic3_base.py` (b) | 3 / 3 | 100.0 | 3 / 3 | 100.0 | 98 |
| `models/moss_image/model_config.py` (b) | 23 / 31 | 74.2 | 30 / 31 | 96.8 | 94 |
| `utils/ema.py` (b) | 51 / 73 | 69.9 | 56 / 73 | 76.7 | 74 |
| `training/progress.py` (c) | 15 / 41 | 36.6 | 40 / 41 | 97.6 | 95 |
| `training/dop.py` (c) | 17 / 19 | 89.5 | 19 / 19 | 100.0 | 98 |
| `finetune/regularisation.py` (c) | 7 / 7 | 100.0 | 7 / 7 | 100.0 | 98 |
| `finetune/resume.py` (c) | 8 / 8 | 100.0 | 8 / 8 | 100.0 | 98 |
| `finetune/extract.py` (c) | 87 / 98 | 88.8 | 97 / 98 | 99.0 | 97 |
| `data/walk.py` (c) | 4 / 4 | 100.0 | 4 / 4 | 100.0 | 98 |
| `data/media.py` (c) | 45 / 59 | 76.3 | 54 / 59 | 91.5 | 89 |
| `utils/batching.py` (c) | 5 / 5 | 100.0 | 5 / 5 | 100.0 | 98 |
| `utils/config.py` (c) | 10 / 12 | 83.3 | 12 / 12 | 100.0 | 98 |
| `models/embedding.py` (c) | 14 / 20 | 70.0 | 16 / 20 | 80.0 | 78 |
| `models/minimax_h3/packing.py` (c) | 123 / 127 | 96.9 | 126 / 127 | 99.2 | 97 |
| `models/minimax_h3/video_frames.py` (c) | 24 / 25 | 96.0 | 24 / 25 | 96.0 | 94 |

(a) Added to the set after the first run; its tests did not change.

(b) Added later, tests only (these cover MOSS-Image finetune and Automagic3 code: no product code was touched).

(c) Added in the second round (pure logic: progress line, DOP, finetune extraction and resume checks, media timing,
packing). Before is the first run on the tests as they were; `video_frames.py` was mapped to the tests that already
pinned it (a first run with a wrong mapping gave 0 / 25). No product code was touched.

All 33 modules: 1157 / 1374 killed (84.2 %) before, 1343 / 1403 (95.7 %) after; the remaining 60 survivors are the
equivalent mutants below (the first 17 modules: 682 / 795 before, 797 / 824 after).

`training/steps.py` was 15 % because `plan_training_steps` and `epoch_order` were only exercised through trainer
runs with a default config. The other gaps were the RoPE theta (10000 against 10001 changes an angle by about
1e-5 radians per position, below the default tolerance, so the tests use positions in the hundreds), the
defaults of `sample` (steps, seed, cfg, device), the batched negative prompt, the bf16 branches of Automagic3's
second moments (never run with a non-trivial state), the bit packing of the sign history and the vote masks,
weight decay of exactly 1, and the batch dimension of the factored moments.

Runtime for the full set: 4 to 5.5 minutes wall time for the first 17 modules, 7.5 minutes for all 33 (4 workers, on a
machine that was also running other work); one module takes 5 s to 1 min, so `--changed` is usually under two minutes. Adding workers helps until the
cores or the memory run out.

## Equivalent mutants

Judged by reading the code. Line numbers are those of the measured commit. Add a row when you accept one.

| Module and line | Mutant | Why it changes nothing |
|---|---|---|
| `training/steps.py` 26 | `num_processes > 1` to `>= 1` | With one process the only valid `process_index` is 0, and `order[0::1]` is the whole order. |
| `models/moss_image/sampling.py` 43 | clamp max `TRAIN_STEPS - 1` to `+ 1` | Sigmas never exceed the largest training sigma (1.0), so the clamp is a guard. |
| `models/moss_image/sampling.py` 44 | `clamp(min=0)` to `min=1` | `lower` only differs at the smallest training sigma, where `lower` and `upper` snap to the same value. |
| `models/moss_image/sampling.py` 45 | `<=` to `<` | A tie needs a sigma exactly half way between two training sigmas. |
| `models/moss_image/sampling.py` 81 | `use_cfg and negative is not None` to `or` | The second operand only narrows the type for mypy; with `cfg == 1` the negative is converted and not used. |
| `models/moss_image/model_math.py` 23 | `cat(..., dim=-1)` to `dim=1` | Always a 2-D tensor. |
| `models/qwen_image_2_1/model_math.py` 41, 42 (x2) | table sizes 8192 and 1024 plus one; `* -1` to `/ -1` | Only positions beyond 8192 text tokens or 1024 negative rows (outside the supported range) read the extra row; the division gives the same values as floats. |
| `models/qwen_image_2_1/model_math.py` 67 | `cat(..., dim=-1)` to `dim=1` | Always a 2-D tensor. |
| `models/flux3/rope.py` 15 | `dim=256` to `257` | `257 // 2 == 256 // 2`, and the embedding is the two halves concatenated. |
| `models/flux3/rope.py` 22, 37 | `dim=-1` to `1`; `ids.shape[-1]` to `shape[1]` | 2-D tensors (`[B, half]`, `[N, 4]`). |
| `models/krea2/rope.py` 16 | `cat(..., dim=-1)` to `dim=1` | Always a 2-D tensor. |
| `optimizers/automagic3_ops.py` 38, 43 | `sum(-1)` to `sum(1)`; `unsqueeze(-1)` to `unsqueeze(1)` | A 2-D `[n, 8]` tensor and a 1-D byte tensor. |
| `optimizers/automagic3_update.py` 104, `automagic3_stats.py` 34 | `device !=` to `==` (and the reverse) | A guard that only moves a scalar between devices; indistinguishable on CPU. Needs a second device to test. |
| `utils/step_planner.py` 75 | `default=0` to `1` | The width is only used by folder rows, and there are none when the default applies. |
| `optimizers/automagic3_resume.py` 29 | `zip(..., strict=False)` to `True` | `param_groups` and the saved `_built` always have the same length (torch refuses a state with another group count first). |
| `optimizers/automagic3_resume.py` 45 | the `"lr"` key of the assignment | The rate was already restored as fp32 under `"lr"` a few lines above, so the cast stores the same tensor. |
| `models/moss_image/model_config.py` 32 | `split(".", 2)` to `3` | `layers.<i>.rest`: element 1 is the index for either limit. |
| `utils/ema.py` 37, 46, 52, 102, 115, 123, 132, 136, 142, 155 (x10) | `zip(..., strict=False)` to `True` | The lists zipped are built together from the same parameters, so they always have equal length. |
| `utils/ema.py` 103, 133, 137, 143 (x4) | `non_blocking=True` to `False` | Only changes how a CUDA copy is queued; on CPU the copy is the same. |
| `utils/ema.py` 108 | `daemon=True` to `False` | The thread is always joined by `wait()` before its result is read; the flag only matters at interpreter exit. |
| `utils/ema.py` 111 (x2) | `chunk = 1 << 20` plus one on either operand | The chunk size is a cache-blocking choice; any size gives the same averages. |
| `training/progress.py` 41 | `zip(..., strict=False)` to `True` | The groups and the rates come from the same optimizer: equal length. |
| `finetune/extract.py` 97 | `full_matrices=False` to `True` | The factors use only the first `k` columns and rows of `u` and `vh`, and `s` is the same. |
| `data/media.py` 42 | `last_time = 0.0` to `1.0` | `last_time` is read only after a frame was pushed, which sets it. |
| `data/media.py` 56 | `<` to `<=` against `end - 1e-06` | A tick time equal to the end minus a millionth of a second is not reachable. |
| `data/media.py` 132, 134, 137 | `copy=False` to `True`; the mono `repeat(wav, 2)` and `wav[:2]` to 3 | A copy has the same values; three repeated rows cut to `[:2]` are the two of the original; the resampler already gives at most 2 channels. |
| `models/embedding.py` 45 | `num_vectors > 1` to `>= 1` | With one vector the extra-token range `range(1, 1)` is empty. |
| `models/embedding.py` 97 (x2), 102 | the defaults `strict=True`, `assign=False`, `recurse=True` | Unused arguments of `nn.Module` signatures (`ARG002`). |
| `models/minimax_h3/packing.py` 80 | `stack(..., dim=-1)` to `1` | Two 1-D tensors: the result is 2-D either way. |
| `models/minimax_h3/video_frames.py` 19 | `frames - 5` to `frames + 5` | Only for clips on the 17k + 5 grid (its documented input) both give `k`; `align_frame_count` produces such counts. |
| `finetune/classes.py` 22 | `"t0"` to another string | `match.lastgroup` is always set after a match; the fallback is unreachable. |
| `models/minimax_h3/int8_rotation.py` 16 (x2) | `value > 1` to `>= 1`; `1` to `2` | `1 % 4` and `2 % 4` are not 0, so the loop stops at either. |
| `models/minimax_h3/int8_rotation.py` 30, 33 | the loop counter starts at 5 or grows by 5 | The loop runs the same number of times for every power of four up to 4^8. |
| `models/minimax_h3/int8_rotation.py` 49 (x2) | `clamp(-127, 127)` to `-128, 128` | The scale maps the largest value to exactly 127, so the clamp never triggers. |
| `models/minimax_h3/int8_rotation.py` 57 | `rows > 16` to `> 17` | At 17 rows the padding is zero rows. |

## Policy

1. **When it must be run.** Before merging a change that touches a target module in
   `maltrainer_quality/mutation_targets.py` or a test file mapped to one: run
   `python -m maltrainer_quality.mutation --changed` and put its summary in the pull request. It is not in the
   default gate, because of its runtime. In CI it is the separate `mutation` job of `quality.yml`, which runs
   the same command on pull requests that touch those paths (and the whole set by hand, `workflow_dispatch`).
   **Decision (2026-10-21): it stays an optional job, not a required check.** It takes 5 to 8 minutes, a
   survivor needs a human to judge (a test to add, or an equivalent mutant to document), and the repository has no
   branch protection to make any check required (see "Enforcement" in the README). Revisit if that changes.
2. **The floor.** Each target has a minimum score in `mutation_targets.py`, set a little below the measured one
   (the measured whole percent minus 2). A module below its floor fails the command and the job. Floors are only
   raised, never lowered, like `fail_under`: when a change raises a score, raise the floor in the same commit.
3. **A surviving mutant.** If it is not equivalent, add an assertion that kills it, in the unit test of that
   module, before the change merges. If it is equivalent, add it to the table above with the reason. Do not edit
   product code to kill a mutant, and do not lower a floor.
4. **New and changed tests of these modules.** Run the mutation on the module you touched. A new assertion has to
   kill something; a test changed to make a failing run pass has to keep the old mutants dead. Tests that only
   run the code without pinning a value are what this catches.
5. **New pure-math modules.** A new module of that kind (a schedule, a RoPE, an optimizer update, a planner) is
   added to `TARGETS` in the commit that adds it: its unit tests, and a floor from its first measured score.
6. **The tool itself.** It has its own unit tests (`tests/maltrainer_quality/test_mutation_*.py`) that run in the
   default gate as ordinary tests.
