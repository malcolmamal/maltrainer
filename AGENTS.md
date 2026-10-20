# Working on maltrainer

Run the quality gate before every commit: `.venv/Scripts/python -m maltrainer_quality` (see "Quality gates" in
`README.md`). It must end with no failed gates.

- **Limits:** 200 lines per Python or shell file, complexity 10 per function. Older code is listed in
  `maltrainer_quality/baselines/` at its size then and may only shrink. After shrinking something, run the three
  `--tighten` commands from the README.
- **Never** add a baseline entry, raise one, or lower the coverage floor (`fail_under`) to make a change pass. Split the
  file or function, or write the test.
- **Bugs:** write the failing test first and check it fails on the old code. In the commit message say which area is
  affected (LoRA or finetune, which model type). Anything that touches the MOSS-Image finetune
  (`train_moss_image_finetune.py`, `maltrainer/finetune/`, `maltrainer/models/moss_image/`) must be called out
  separately: it is in daily use, and the exact-resume tests in `tests/trainers/moss_image/` must keep passing.
- **Tests** run real trainer code on tiny CPU models; only loaders are faked. Keep them annotated like the product code.
- Shared helpers live in `maltrainer/` (optimizers, finetune, data, utils); reuse them instead of copying between
  trainers.
