# LoRA Checkpoint Metadata (MiniMax H3)

Every LoRA checkpoint written by `train_minimax_h3_lora.py` embeds the training
configuration and the resolved run details in its safetensors metadata header. A checkpoint
is therefore self-describing: months later you can recover the exact config that produced
it, or diff two files to see what actually differed between two runs.

Inspect with `scripts/inspect_lora_metadata.py` (see [Inspecting](#inspecting)).

## Why

The hyperparameters in a YAML file are not the whole story. Several values that decide how
a run behaves are only resolved at runtime:

- the sigma shift `krea2_shift` computes from the latent size (~1.65 at 512px, ~1.79 at 1024px)
- the dtype the adapters actually trained in, which is not always what the config asked for
- the optimizer arguments that survived the installed `prodigyopt` signature, since keys
  like `slice_p` and `growth_rate` are not in every release
- which revision of the trainer ran, which matters because this trainer's numerics have
  changed materially between runs

Re-deriving those from a YAML after the fact is guesswork. They are recorded explicitly.

## What is stored

safetensors metadata is a flat `Dict[str, str]`, so every value is stringified and anything
structured is JSON-encoded. Three groups of keys:

### `ss_*` — flat scalars

Follows the kohya / musubi-tuner convention, so tools that already look for `ss_` keys show
something useful.

| Key | Meaning |
| --- | --- |
| `ss_base_model_version` | `minimax_h3_fl2va` |
| `ss_network_module` | `maltrainer.models.minimax_h3.lora` |
| `ss_network_dim`, `ss_network_alpha` | LoRA rank and alpha |
| `ss_network_args` | JSON: `target_modules`, `prefix`, adapter `dtype` |
| `ss_optimizer` | optimizer name as configured |
| `ss_learning_rate`, `ss_weight_decay`, `ss_lr_warmup_steps` | as configured |
| `ss_num_epochs`, `ss_batch_size`, `ss_gradient_accumulation_steps` | run shape |
| `ss_max_grad_norm`, `ss_mixed_precision`, `ss_seed` | |
| `ss_resolution` | dataset resolution |
| `ss_timestep_sampling` | `krea2_shift`, `uniform`, or `logit_normal` |
| `ss_sigmoid_scale` | logit-normal width for `krea2_shift` |
| `ss_discrete_flow_shift` | the fixed `timestep_shift`; used only by `uniform`/`logit_normal` |
| `ss_transformer`, `ss_vae`, `ss_text_encoder` | base model **filenames** only |

### `maltrainer_config` — the full config

The complete YAML config as JSON. It round-trips byte-identical, so
`--config -o recovered.yaml` produces a file you can pass straight back to the trainer.

### `maltrainer_*` — resolved run details

| Key | Meaning |
| --- | --- |
| `maltrainer_metadata_version` | schema version of this metadata block |
| `maltrainer_architecture`, `maltrainer_trainer` | `minimax_h3`, script name |
| `maltrainer_git_commit` | trainer revision, suffixed `-dirty` if the tree was modified |
| `maltrainer_torch_version` | |
| `maltrainer_started_at`, `maltrainer_saved_at` | run start and this checkpoint's write time |
| `maltrainer_resolved_sigma_shift` | the shift actually used, e.g. `1.648721` |
| `maltrainer_latent_size` | e.g. `32x32` |
| `maltrainer_lora_dtype` | dtype adapters trained in (`float32` unless overridden) |
| `maltrainer_save_dtype` | dtype the file was written in (`bfloat16` by default) |
| `maltrainer_trainable_parameters`, `maltrainer_lora_layers` | adapter size |
| `maltrainer_optimizer_resolved` | JSON: the kwargs the optimizer was actually built with |
| `maltrainer_int8_grad_precision` | `bf16`, `fp32`, or `int8` backward through the frozen base |
| `maltrainer_steps_per_epoch`, `maltrainer_planned_total_steps` | |
| `maltrainer_dataset_samples` | images x repeats |

Legacy keys `step`, `rank`, `alpha`, `architecture` are still written for anything that
already read them.

## Inspecting

```bash
# summary of how a checkpoint was trained
python scripts/inspect_lora_metadata.py output/minimax_h3_lora/minimax_h3_lora_step_2250.safetensors
```

Only the header is read, so this is instant regardless of file size.

```bash
# recover the config, then reproduce or fork the run
python scripts/inspect_lora_metadata.py MODEL.safetensors --config -o recovered.yaml
python train_minimax_h3_lora.py --config recovered.yaml
```

```bash
# what ACTUALLY differed between two runs
python scripts/inspect_lora_metadata.py A.safetensors --diff B.safetensors
```

`--diff` ignores the config blob and timestamps and lists only genuine setting
differences. This is the most useful mode when comparing runs: it catches the variable you
forgot you changed, which is easy to do when epoch counts and schedules move together.

```bash
# everything, for grepping or scripting
python scripts/inspect_lora_metadata.py MODEL.safetensors --all
python scripts/inspect_lora_metadata.py MODEL.safetensors --json
```

`--help` documents all of the above.

## Disabling

```yaml
output:
  path: "./output/minimax_h3_lora"
  embed_config: false   # omit maltrainer_config; ss_* scalars are still written
```

Worth doing if a LoRA will be shared publicly: the embedded config contains absolute local
paths for the dataset and base models.

`output.dtype` (default `bfloat16`) controls the saved tensor dtype, independent of the
`float32` adapters used during training. At rank 32-64 across H3's dimensions this roughly
halves file size with no practical quality cost.

## Caveats

- Checkpoints written before this feature carry no metadata. The inspector reports
  `no metadata` rather than failing.
- `maltrainer_planned_total_steps` is the planned count from the config. A run stopped early
  still reports the plan; the `step` key is what the file actually reached.
- Programmatic access: `maltrainer.utils.lora_metadata` exposes `read_metadata(path)` and
  `extract_config(metadata)`.
