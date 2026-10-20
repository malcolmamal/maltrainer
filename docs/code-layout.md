# Code layout

Where things live, and the rules that keep them there. For the quality gate see the "Quality gate" section of the
[README](../README.md); for the system view see the [architecture](architecture.md).

## The shape of a run

```
train_<model>_<kind>.py            thin entry point: parse args, load the YAML config, call the trainer
        |
maltrainer/trainers/<model>/       the model's training pipeline (data, model loading, one step, loop, previews)
        |  builds on
maltrainer/training/               what every trainer shares (session, steps, loop, optimizer, saving, CLI)
maltrainer/models/<model>/         the network, its VAE and text encoder, loaders, LoRA layers
maltrainer/optimizers/             Automagic3 and the Prodigy builder
maltrainer/data/                   dataset scanning, bucketing, caches
maltrainer/finetune/               what full finetuning adds (selection, EMA/prior terms, resume, extraction)
maltrainer/utils/                  config, logging, previews and the preview menu, monitor, EMA, file helpers
```

The root `train_*.py` files stay small on purpose (200 lines at most, like every Python file). Anything reusable
moves into `maltrainer/`.
The package layers are a guide, not a strict hierarchy: `models`, `finetune` and `utils` import each other in a few places
(see the import table in [architecture.md](architecture.md)).

## Top level

| Path | What it is |
|---|---|
| `train_<model>_lora.py`, `train_krea2_finetune.py`, `train_moss_image_finetune.py`, `train_zimage_embedding.py` | One entry point per trainer: FLUX.3 [klein], Krea 2, MOSS-Image, MiniMax H3, Qwen-Image 2.1, Z-Image (LoRA and textual inversion), and the two full finetunes. Each also has a `run_*.bat` launcher. |
| `check_weights.py`, `compare_loras.py`, `inspect_lora.py` | Small root-level LoRA inspection tools. |
| `scripts/` | Command-line tools that are not trainers: merge, resize and extract LoRAs, inspect metadata, inference, the Qwen 2.1 benchmark, `train_from_template.py`. Helper modules sit next to the script that uses them (`scripts` has no package). |
| `backend/` | The web UI's FastAPI service (see below). |
| `frontend/` | The Angular web UI. |
| `templates/` | YAML config templates the UI and `train_from_template.py` start from. |
| `comfyui-zimage-embedding/` | A ComfyUI custom node for the Z-Image embeddings (ComfyUI's own naming rules apply there). |
| `maltrainer_quality/` | The quality gate runner (`python -m maltrainer_quality`), with its gates as modules. |
| `tests/` | Mirrors the code: `tests/maltrainer/`, `tests/trainers/<model>/`, `tests/backend/`, `tests/scripts/`, `tests/maltrainer_quality/`. |
| `docs/` | User and design documents. |
| `work/`, `output/` | Git-ignored: internal notes, training outputs. Tests must never write to `output/`. |

## `maltrainer/`

### `training/`: shared trainer machinery
`session.py` (accelerator, seeds, device), `steps.py` (step and epoch planning), `loop.py` (`StepLoop` with hooks),
`image_lora.py` (`ImageLoRATrainer`, the base of the FLUX.3, Krea 2 and MOSS-Image LoRA trainers), `optimizer.py`,
`progress.py`, `saving.py`, `sigma.py` (timestep sampling), `dop.py` (differential output preservation), `cli.py`.
H3, Qwen 2.1 and Z-Image have loops that differ too much to share the base; they reuse only the small pieces.

### `trainers/<model>/`: one model's pipeline
Not every trainer has a package here. `trainers/` exists for MiniMax H3, MOSS-Image, Qwen-Image 2.1, Z-Image and the
two finetunes' parts. The FLUX.3 [klein] and Krea 2 LoRA trainers are classes on `ImageLoRATrainer` that live in
their root `train_*.py` files (still under 200 lines), and the finetune entry points import the LoRA script they
extend (`train_krea2_finetune.py` -> `train_krea2_lora.py`, `train_moss_image_finetune.py` -> `train_moss_image_lora.py`).
Modules by role: `data` (what is loaded), `model` (loading and wiring), `step` (one forward/backward), `loop`,
`optimizer`, `previews`/`sample`, plus model specifics (`audio`/`visual` for H3, `pipeline` for the Z-Image run
order). The finetunes are split into `finetune_*` modules:

- `trainers/moss_image/finetune_*`: mixins (`data` -> `prior` -> `state` -> `checkpoint`, listed ahead of the LoRA
  trainer in the class bases) plus functions (`model`, `loop`, `resume`). A stub that must not shadow a later
  class in the method resolution order is declared under `if TYPE_CHECKING:`.
- `trainers/krea2/finetune_*`: `host`, `model`, `loop`, `save`.

### `models/<model>/`: the networks
Each model package follows the same split, so a file's name tells you its content:

| Part | Typical modules |
|---|---|
| Transformer | `model.py` (the module), `model_config.py`, `model_blocks.py` / `layers.py`, `model_math.py` / `rope.py` |
| VAE | `vae.py` (wrapper and loader), `vae_blocks.py`, `vae_stages.py`, `vae_networks.py`, `vae_constants.py` |
| Text encoder | `text_encoder.py` (encoder and loader), plus the model's language-model parts (`qwen35_*` for MOSS-Image) |
| Loading | `loading.py` / `checkpoint.py`: read a single ComfyUI-format file, map keys, handle INT8/fp8 |
| LoRA | `lora.py`, `lora_layer.py`, `lora_assistant.py` (training adapters merged while loading) |

Shared: `models/lora.py`, `models/embedding.py`, `models/scheduler.py`, `models/comfy_fp8*.py` (ComfyUI fp8 files),
and `models/minimax_h3/int8_*.py` + `comfy_quant*.py` (INT8 ConvRot and NVFP4 layers, also used by the FLUX.3,
Qwen 2.1 and MOSS-Image loaders). `supra2_img/` is the Supra2 model used by the Z-Image side.

A module that was split keeps its old name and re-exports every public name (`__all__`), so old import paths keep
working. Tests patch a name in the module where the code looks it up; moving a name means moving its patch.

### `optimizers/`
`automagic3*.py`: the optimizer, split into mixins (`update`, `stats`, `resume`, `ops`, `base`).
`prodigy.py`: `build_prodigy`, shared by the trainers that offer Prodigy.

### `data/`
`dataset.py` and one dataset per model (`flux3_dataset`, `krea2_dataset`, `qwen_dataset`, `minimax_dataset` +
`minimax_scan`), `walk.py` (the one directory walk: skips hidden folders), `media.py` (images and video frames),
`disk_cache.py` (latent and embedding cache, shared by the MOSS trainers).

### `finetune/`
Full-finetuning building blocks: `selection` (which parameters train), `rotation` (Krea 2 fp8 window rotation),
`prior`/`classes`/`regularisation`/`repeats` (class-prior preservation), `resume` (exact-resume checks), `extract`
(finetune -> LoRA by SVD), `notes`.

### `utils/`
`config.py`, `logging_utils.py`, `fsutil.py` (rename with retry for Windows), `previews*.py` and `preview_menu.py`
(the live preview and its keyboard menu), `gallery.py`, `step_planner.py` (`--suggest-steps`), `ema.py`,
`train_monitor.py`, `lora_metadata.py`, `batching.py`, `keyboard.py`/`console_keys.py`, `compile.py`.

## `backend/`
`app.py` (the `backend.app:app` entry), `specs` (what each trainer accepts), `registry`, `files` (dataset and file
access, path-safe), `plan` + `guidance` (the `/api/plan` step planner), `training` (job processes), `streams`
(log/progress streaming), `job_routes`, `schemas`.

## Rules that shape the layout
- 200 lines per Python or shell file, complexity 10 per function, mypy strict over the tree: the gate enforces them,
  so the structure above is what it takes to stay inside.
- A change that moves code must not change behaviour. The refactors in this tree were proven by running the old and the
  new code on seeded tiny models and comparing tensors, state dicts, saved files and logs byte for byte.
- Tests run the real trainer code on tiny CPU models and fake only the loaders (`tests/trainers/<model>/`).
