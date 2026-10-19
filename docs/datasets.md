# Datasets: layout, captions, loading

This covers the image trainers that share `maltrainer/data/minimax_dataset.py`'s scanner:
MOSS-Image (LoRA and finetune), Krea 2 and FLUX.3 [klein]. Numbers marked MOSS are for
MOSS-Image; the others differ only in the bucket grid and their text encoder.

## Folder layout

`dataset.path` names a folder. The trainer collects every image under it, **subfolders
included**, at any depth:

```
datasets/my_set/
  anna/
    001.jpg
    001.txt        <- caption for 001.jpg
    002.png
    002.txt
  ben/
    beach.webp
    beach.txt
  both/
    party.jpg
    party.txt
```

- Formats: `.jpg`, `.jpeg`, `.png`, `.webp`, in any letter case. Everything else, including the
  `.txt` files, is ignored as an image.
- Order: sorted by full path, then shuffled again every epoch.
- Subfolders are only for organisation. Folder names add nothing to captions or weighting; every
  image counts once.
- An image that fails to load is skipped with a warning; the run continues.
- `dataset.repeats` repeats the whole set within each epoch. It is global, not per folder (see
  [Balancing](#balancing)).

## Captions

Each image's caption comes from the first of these that applies:

1. **Sidecar file**: `<same name>.txt` in the same folder (`001.jpg` → `001.txt`), UTF-8. A
   non-empty file is used **exactly as written**. The trigger word is **not** added to it, so
   write the trigger into the caption yourself.
2. **Fallback**: if the `.txt` is missing or empty, the caption is `dataset.default_caption`.
   The trigger is prepended unless the default caption already contains it. The templates set
   `default_caption: "a photo of <trigger>"`.
3. With neither a default caption nor a trigger, the caption is empty.

### Caption prefix: `prefix.txt`

A `prefix.txt` puts the same words in front of every caption below it. `[folder]` stands for
the name of the folder that directly holds each image:

```
datasets/
  prefix.txt            "photo of [folder]"
  people/                     <- dataset.path
    feliciaday/
      a.jpg  a.txt "smiling, red hair"   -> "photo of feliciaday, smiling, red hair"
      b.jpg  (no .txt)                   -> "photo of feliciaday"
    ben/
      prefix.txt        "portrait of [folder]"
      d.jpg  d.txt "on a beach"          -> "portrait of ben, on a beach"
```

- **Which file applies:** for each image, the nearest `prefix.txt`. The search starts in the
  image's own folder and walks up to **one level above** `dataset.path`.
  - So `datasets/prefix.txt` covers `dataset.path: datasets/people`, and also
    `datasets/feliciaday` when that folder is trained on its own.
  - A closer `prefix.txt` replaces a farther one. An **empty** `prefix.txt` switches the prefix
  off for its folder and everything below it.
- **Joining:** the prefix comes first, then ", ", then the image's `.txt` caption. A trailing
  comma in `prefix.txt` is dropped, and its line breaks become spaces. A caption that already
  starts with the prefix is left alone.
- **No `.txt`:** an image without one gets the prefix alone. The prefix names the subject, so
  `default_caption` and the trigger are not added on top.
- **Nested folders:** `[folder]` is the image's own folder. `anna/close/c.jpg` becomes "photo of
  close". Keep each person's images directly in their folder, or put a `prefix.txt` with the
  name written out in `anna/`.
- **Evaluation:** every run reads the prefix fresh. Edit `prefix.txt` and the next run picks up
  the new captions. Only the changed captions are re-encoded; latents are untouched. A resumed
  finetune sees the change as a changed dataset and starts a fresh epoch.

Without any `prefix.txt`, captions behave exactly as described above.

So a folder with no `.txt` files and no `prefix.txt` trains every image on the same caption,
`a photo of <trigger>`.
That is fine for one person with a uniform set. Per-image captions do better as soon as images
differ (outfit, setting, framing), and they are required for more than one person.

`train.caption_dropout` (0.05 in the templates) swaps the caption for the empty prompt on that
share of steps. ComfyUI's CFG uses the empty prompt as its negative, so this keeps the
unconditional branch in step with the trained one.

### How captions are encoded (MOSS)

MOSS-Image conditions on **Qwen3.5-9B**, an LLM, not CLIP. Each caption is wrapped in the chat
template and encoded at its true length, with no padding and no 77-token limit. The cap is 4096
tokens, and longer captions are cut with a warning. The model was pretrained on long, detailed
natural-language captions, so write sentences:

```
ohwx anna, a woman with shoulder-length red hair, smiling, in a blue denim jacket, standing on a
city street at dusk, photographed from the waist up
```

What to put in a caption: describe what should stay **changeable** (clothes, background,
lighting, pose, framing). Leave out what the trigger should **absorb** (face, identity). Whatever
is never mentioned gets bound to the trigger.

## Two or more people

One dataset can teach several people. The quickest setup is one folder per person, named after
their trigger, plus a `prefix.txt` with `photo of [folder]`. Each image's `.txt` then only
describes the scene.

Group shots need their own captions naming both people. Put them in a folder with its own
`prefix.txt`, even an empty one, or write the full caption per image.

Written out by hand, give each person their own trigger and caption every image with `.txt`
files:

| Image shows | Caption starts with |
|---|---|
| Anna alone | `ohwx anna, a woman ...` |
| Ben alone | `sks ben, a man ...` |
| Both | `ohwx anna on the left and sks ben on the right, ...` |

- Use **distinct, unrelated** triggers. Names that share a token, or are close in spelling, bleed
  into each other. A rare token plus the name (`ohwx anna`) works well.
- Images with **both** people are what teach the model to keep them apart. Say who is where
  (left/right, foreground, "the man in the red shirt").
- In the config, `dataset.trigger` and `default_caption` matter only for images without a `.txt`.
  With several people, every image should have one. Set `default_caption: ""` and
  `trigger: ""`, so a forgotten caption trains on nothing rather than on the wrong person.
- Each person needs roughly the image count you would use for them alone (see below).
- Capacity: several people in one rank-16 LoRA is untested here. If they blend, raise
  `lora.rank` (32-64) or use a [full finetune](moss-image-finetuning.md), which trains every
  weight.

## Balancing

Every image is one sample per epoch. If Anna has 60 images and Ben 20, Anna gets three times the
training. To even it out:

- copy Ben's images (with their `.txt`) a second and third time under new names, or
- trim Anna's set to the best 20-30.

`dataset.repeats` cannot fix an imbalance: it repeats everything equally.

## How big

What works for likeness (MOSS LoRA runs so far): **20-30 good images per person**. The
feliciaday run used 23 at 512 px. Variety matters more than count: different angles,
lighting, outfits, distances. A few close-ups and a few full-body shots beat 100 near-identical
selfies.

Larger sets are fine. Limits come from RAM and caching time, not the trainer:

- **RAM.** Every latent and every distinct caption's embedding is cached in CPU RAM for the whole
  run. On MOSS a latent is 128 × (W/16) × (H/16) fp32: 0.5 MB at 512 px, 2 MB at 1024 px. An
  80-token caption is about 0.65 MB. So 1,000 images at 1024 px with unique captions use about
  2.7 GB, and 10,000 about 27 GB.
- **Encoding time** applies only to images and captions not already in the disk cache (below).
  - VAE: ~25 images/s at 512 px.
  - Text encoder: ~3 captions/s measured on short captions. So 1,000 unique captions take
    5 minutes or more; identical captions are encoded once.
- **Steps.** One epoch is `images × repeats` steps (batch size 1). For large sets, set
  `train.max_steps` rather than counting epochs.

## Encoder cache (MOSS-Image)

Every image is VAE-encoded and every caption is text-encoded before training. Both encoders are
then unloaded, so the DiT has the GPU to itself. The results are saved next to the images: each
folder that holds images gets a `.maltrainer_cache/` with the latents of its own images and the
embeddings of their captions. The empty prompt used for caption dropout goes in the dataset
folder itself.

```
datasets/my_set/
  .maltrainer_cache/text/...          <- the empty dropout prompt
  anna/
    001.jpg  001.txt  ...
    .maltrainer_cache/latents/...     <- anna's images
    .maltrainer_cache/text/...        <- anna's captions
  ben/
    .maltrainer_cache/...
```

The next run, a LoRA or a finetune and including `resume_from`, reads them back. It loads
neither encoder unless something is new: only new or changed images and captions are encoded.

Each entry's key hashes everything that shapes it:
- **Latents:** the image's bytes, `resolution`, `aspect_mode` and the VAE file (name + size).
- **Captions:** the caption text and the text-encoder file.

So:

- Copy, move or rename a folder and its cache still matches. The key is the content, not the
  path.
- Edit an image or a `.txt`, or change `resolution`, and only those entries are re-encoded. The
  old entries stay on disk unused; delete a `.maltrainer_cache` folder to clear it. The trainer
  rebuilds whatever it needs.
- Delete a folder and its cache goes with it.
- Size per entry: a latent is 0.5 MB at 512 px and 2 MB at 1024 px; an 80-token caption is about
  0.65 MB.

Cached tensors are bitwise identical to freshly encoded ones (tested). Set `dataset.cache: false`
to encode in memory only, as before.

## Resolution and buckets

- `dataset.resolution` (512 in the templates) is an **area** target, not a side length.
- `aspect_mode: bucket` keeps each image's aspect ratio. The image is scaled so its area is about
  `resolution²`, each side is rounded down to the model's grid (16 px for MOSS and FLUX.3), then
  centre-cropped by at most a few pixels.
  - A 3:2 photo at `resolution: 512` trains at 624 × 416.
  - A square photo trains at 512 × 512.
- `aspect_mode: square` centre-crops every image to `resolution × resolution` instead, so
  non-square images lose their edges.
- Small images are scaled **up** to the target. Upscaled low-res sources teach softness, so
  prefer sources at least as large as the bucket.
- The log lists the sizes used at the start of each run:
  `Training sizes (w x h: images): 512x512: 23`.

Batch size is always 1, so mixed aspect ratios never need to be grouped. Each step uses one image
at its own size, with its caption at its own length.
