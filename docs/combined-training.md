# Combined training: `scripts/train_combined.py`

A combined run trains one dataset with several optimizers, one after another, and merges the
LoRAs into one file: a "LoRA soup". Each member is an ordinary training config (same model and
dataset, its own optimizer); the merge is exact, done by [`merge_loras.py`](merge-loras.md) at a
strength per member, so the soup applies what stacking the members at those strengths does.

Shrinking the soup to a lower rank is a separate step, [`resize_lora.py`](resize-loras.md).
Stabilis does both for you (see below).

## Quick start

```bash
python scripts/train_combined.py --config config/my_soup.yaml
```

```yaml
trainer: train_moss_image_lora.py          # every member trains with this script
members:
  - config: config/train_moss_image_ana_adamw.yaml
    strength: 0.7
  - config: config/train_moss_image_ana_automagic3.yaml
    strength: 0.7
  - config: config/train_moss_image_ana_prodigy.yaml
    strength: 0.7
output:
  path: ./output/moss_image_ana_combined
  name: moss_image_ana_v1_soup
  dtype: float32                           # optional; float32 keeps the merge exact
```

Then, for example:

```bash
python scripts/resize_lora.py output/moss_image_ana_combined/moss_image_ana_v1_soup.safetensors
```

## How it runs

- **One member at a time.** Each config trains with `python <trainer> --config <member>`. Its LoRA
  is the `<output.path>/<output.name>.safetensors` the config names, and it must be written by
  this run (an older file of that name does not count).
- **A failure stops everything.** If a member exits non-zero, or finishes without writing its
  LoRA, the run stops there and nothing is merged. The LoRAs already trained stay where their
  configs put them, so you can merge them by hand.
- **The merge** is written to `<output.path>/<output.name>.safetensors` and its full path is
  printed last, on a line starting `Merged LoRA:`. An existing merge needs `--force`.
- **Passing options to every trainer:** anything after `--` goes to each trainer run, e.g.
  `scripts/train_combined.py --config x.yaml -- --dry-run`.
- Relative paths are from the current directory, as for training. Keys other than `trainer`,
  `members` and `output` are ignored (Stabilis writes a `publishName` it reads itself).

## Strengths

Two or three LoRAs of the same person, each trained to full strength, are too strong stacked at
1.0. The defaults Stabilis uses: **0.95 each for two members, 0.7 each for three**. Load the soup at
strength 1.0; the member strengths are already folded in.

## From Stabilis

The Training page offers three families under MalTrainer, each with one template per optimizer and
a **combined** one (its Automagic3 template is where the family starts):

| Family      | Automagic3 | AdamW 8-bit | Prodigy    | Copied into ComfyUI under | LoRA name              |
| ----------- | ---------- | ----------- | ---------- | ------------------------- | ---------------------- |
| MOSS-Image  | 20 epochs  | 60 epochs   | 30 epochs  | `loras/Moss/celebs`       | `moss_image_<person>_v1` |
| Krea 2      | 70 epochs  | 100 epochs  | 100 epochs | `loras/Krea2/celebs`      | `krea2_<person>_v1`      |
| Qwen 2.1    | 70 epochs  | 70 epochs   | 30 epochs  | `loras/Qwen/celebs`       | `qwen21_<person>_v1`     |

Templates are named `<family>_template-automagic3.yaml`, `-adamw8bit.yaml`, `-prodigy.yaml` and
`-combined.yaml`; a single run saves `<name>_v1_automagic3` / `_adamw` / `_prodigy`. Picking combined
shows a panel listing the three optimizers with a strength each (at least two ticked). Stabilis then:

1. writes each ticked member's config, exactly as for a single run, and the combined config above;
2. runs `scripts/train_combined.py --config <it> --force` as one queued job;
3. resizes the merge to rank 8 with `scripts/resize_lora.py` (always, for a combined run);
4. copies the resized file into the folder above under the LoRA name, raising the version when that
   name is taken.

A single run is copied the same way (resized first when "Resize LoRA" is on). Krea 2 is only copied
when MalTrainer trained it; a OneTrainer Krea 2 run is not touched.

The MCP `training_start` tool takes the same choice as a `combine` list of
`{ templatePath, strength }`.
