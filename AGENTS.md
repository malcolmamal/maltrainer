# Working on maltrainer

Run the quality gate before every commit: `.venv/Scripts/python -m maltrainer_quality` (see "Quality gates" in
`README.md`). It must end with no failed gates.

- **Limits (hard, no exceptions list):** 200 lines per Python or shell file, complexity 10 per function, no mypy errors
  in the checked tree.
- **Never** loosen a limit, add a `type: ignore` to dodge a real error, or lower the coverage floor (`fail_under`) to make
  a change pass. Split the file or function, or write the test.
- **Bugs:** write the failing test first and check it fails on the old code. In the commit message say which area is
  affected (LoRA or finetune, which model type). Anything that touches the MOSS-Image finetune
  (`train_moss_image_finetune.py`, `maltrainer/finetune/`, `maltrainer/models/moss_image/`) must be called out
  separately: it is in daily use, and the exact-resume tests in `tests/trainers/moss_image/` must keep passing.
- **Tests** run real trainer code on tiny CPU models; only loaders are faked. Keep them annotated like the product code.
- **Mutation testing** covers the pure-math modules (`maltrainer_quality/mutation_targets.py`). If a change touches
  one of them or a test mapped to it, run `.venv/Scripts/python -m maltrainer_quality.mutation --changed` before
  merging: a module below its floor fails, a surviving mutant that is not equivalent gets an assertion (never a
  product-code edit), floors are only raised. It is not in the default gate. Rules and the scores:
  `docs/mutation-testing.md`.
- Shared helpers live in `maltrainer/` (optimizers, finetune, data, utils); reuse them instead of copying between
  trainers.

## Sentinel review (optional, local)

Some machines have the Sentinel review skill installed under `work/sentinel/` (git-ignored, never committed).

- **If `work/sentinel/SKILL.md` exists, obey it.** For every change it covers (bug fixes, behaviour-changing
  refactors, API or configuration changes, tests meant to prove behaviour) run its review loop before calling the task
  done, apply its P0 and in-scope P1 findings, rerun the affected checks, and give its attestation in the final
  response. Use `work/sentinel/CHECKLIST.md` as the repository checklist, `work/sentinel/config.yaml` as the
  configuration (instead of the skill's default `.sentinel/config.yaml`) and `work/sentinel/reviews/` for reports.
  Do not create a root `.sentinel/` folder.
- **If it is not there, do nothing about it:** no error, no warning, no mention, and do not try to install it. The
  skill may be absent on other machines; the quality gate above is the requirement everywhere.
