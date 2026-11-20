# Krea 2: several subjects in one LoRA (Fizgig LoRA Isolation)

[← Krea 2 LoRA training](krea2-lora-training.md) · config: `templates/examples/train_krea2_lora_isolation.yaml`

One LoRA can hold up to four people (or animals, things, styles), and each stays itself. Put two of
them in one picture, or one of them next to a person the model already knows, and nobody's face leaks
onto anyone else. Without a trigger word you still get the model's own people.

This guide covers what the method does, how to lay out and caption the dataset, how to configure the
run, how to prompt the result and what to look for in the log. It follows Fizgig's
`LORA_ISOLATION.md` and `MULTI_PERSON.md` (Fizgig's tests were on Krea 2) and says where this
trainer differs.

## 1. Why a normal LoRA leaks

A normal character LoRA teaches the model "this face goes with the word `zed`". But the face also
attaches to the plain class word every caption shares ("a woman", "a man"). So afterwards strangers in
the background, the friend standing next to your subject, and "a woman" with no trigger all drift
towards the subject. With two subjects trained in one LoRA, they also blend into each other.

Fizgig's Krea 2 test: two LoRAs, same photos, settings, seed and epoch; one difference, isolation.
Rendering 50 pictures of a stranger beside the subject, the normal LoRA made the stranger look like the
subject in 48 of 50; the isolated one in 0 of 50, at the same likeness.

## 2. The four layers

| Layer | What it does | Config |
|---|---|---|
| **Joined trigger** | `zwxem, a woman` is read as `zwxemwoman` while captions load. The class word no longer is the plain token every other woman uses, so the likeness has no bridge to other people. Files on disk are never changed | `isolation.fuse_triggers` |
| **DOP** (differential output preservation, from ai-toolkit) | Each step also runs the same picture's caption with every joined word replaced by its class phrase (`a woman`), and holds the LoRA's prediction there to the base model's. Without a trigger the LoRA learns to change nothing | `train.dop`, `dop_every`, `dop_multiplier` |
| **Several identities at once** | Each subject has its own folder and trigger. Two faces learned side by side teach "a face belongs to a trigger, not to the class". Even a one-person LoRA does better with a second person in it | `isolation.subjects` |
| **Side-by-sides** | A photo of the subject next to a stranger in one wide picture, trained at a higher rate on its own steps. This teaches "this one is the subject, that one stays a stranger" inside one picture, which is where a normal LoRA leaks | `isolation.side_by_side` |

Two helpers: `balance` repeats the smaller folders so every subject trains about as often as the
largest, and masks (section 7) keep the LoRA from learning backgrounds and clothes.

## 3. Preparing the dataset

### 3.1 One folder per subject. There is no "shared" folder.

```
datasets/
  zwxem/                 <- subject 1: every photo that shows this person
    zwxem_001.jpg
    zwxem_001.txt
    ...
  qrxt/                  <- subject 2
    qrxt_001.jpg
    qrxt_001.txt
    ...
  papercut/              <- optional subject 3: a style (kind: style)
```

- **Unique pictures** (only that person) go in that person's folder.
- **Shared pictures** (two or more of your subjects in one photo) go in **one** of the folders, either one.
  Name every subject in the caption with its own trigger (section 4.3). Whichever folder holds the
  photo, every subject's trigger in the caption is joined. A shared photo counts towards its folder's
  size for balancing, so spread them around if you have many.
- **Photos with other people** (not your subjects): keep them in the subject's folder, and describe the
  others plainly without a trigger (`zwxem, a woman, standing next to a man in a grey coat`).
- Do not put a subject's trigger in a caption unless that person is actually in the photo.
- One trigger per folder. Two folders may not share a trigger or a path.
- A mask file next to a photo is never a training image (`photo-masklabel.png`).
- Sub-folders inside a subject's folder are scanned too; hidden folders (`.name`) are skipped.

### 3.2 How many pictures, what kind

Fizgig's reference run: 36 photos of subject 1, 53 of subject 2 and 8 pictures of a paper-cut style,
balanced. The numbers are not magic; the useful rules:

- 20 to 60 good photos per person. More variety of light, angle, expression and clothing beats more
  near-duplicates.
- Mix close-ups and body shots, but caption what is visible (section 4.4).
- Photos of the same person from the same shoot in the same outfit teach the outfit. Vary it, or use
  masks.
- Rough balance is fine; `balance: true` repeats a smaller folder to match the largest (rounded:
  36 vs 53 stays 1:1; 10 vs 50 repeats the small folder 5x). A subject with 8 pictures against 50 trains
  its pictures 6x per epoch, so expect overfitting on it sooner. Add pictures when you can.
- Resolution: `dataset.resolution` is an area target (512 means about 512x512). Fizgig tested 0.25 MP.
- Image Prep's face crops: files named `FaceCrop_*` are used first for the side-by-sides (section 6).

### 3.3 Who can be a subject

| `kind` | What it is | Joined trigger | DOP caption | Side-by-sides |
|---|---|---|---|---|
| `person` | a person (woman, man, girl, boy, lady, guy, person, child, kid, baby, teenager) | yes (`zwxemwoman`) | `a woman` | yes |
| `thing` | an animal or object (dog, cat, puppy, kitten, horse, or any word you give in `what`) | yes (`rexodog`) | `a dog` | no |
| `style` | a look (paper-cut art, a film stock) | no, trigger kept as written | trigger dropped | no |
| `auto` (default) | `person` when the class word is a person word, otherwise `thing` | | | |

A style is called only when its word is used: DOP's trigger-free pass has no style word, so the model's
own look returns without it.

## 4. Captioning

### 4.1 The rule

Start every caption with the trigger, a comma, and what the subject is:

```
zwxem, a woman smiling in a park, close-up, wearing a white top
qrxt, a man standing on a beach, wearing a grey jacket
rexo, a dog lying on a sofa
```

When training reads it: `zwxemwoman smiling in a park, close-up, wearing a white top`. The file is
unchanged. The run log prints each subject and what its captions are read as.

The trigger is a stand-in for the person only; **describe everything else** (clothes, setting, pose,
light, expression). Whatever you leave out tends to be learned as part of the person.

### 4.2 What gets joined

For each subject the trainer works out what it is, then adds these rewrites, in order:

1. `zwxem, a woman` becomes `zwxemwoman`
2. `zwxem a woman` becomes `zwxemwoman`
3. a bare `zwxem` (a whole word, any case) becomes `zwxemwoman`

The class phrase comes from `what` in the config if you set it, else from your captions: the most
common `"<trigger>, a <up to three words> <class word>"` opening, else the class word the subject's
captions mention most. If it can't tell, the run stops and says which subject to give a `what`.

Describing words work, but read the result: `zwxem, a young woman smiling` becomes
`zwxemwoman, a young woman smiling` (the trigger is joined, the rest is kept). That is fine. DOP removes
the repeated "a young woman" after the swapped trigger on its side. If you want clean captions, write
`zwxem, a woman, young, smiling`.

### 4.3 More than one subject in a picture

Use every subject's own trigger, each followed by what it is:

```
zwxem, a woman, and qrxt, a man, laughing at a table
```

It reads as `zwxemwoman, and qrxtman, laughing at a table` (both joined, whichever folder the file is in),
and DOP's caption becomes `a woman, and a man, laughing at a table`. Name the others, not your subjects,
plainly: `zwxem, a woman, and a tall man in a grey coat on her left`.

### 4.4 Captions that serve the face

- Name pieces you want to stay changeable (hair colour, clothes, glasses, make-up) in the caption, so they
  do not get stuck to the identity.
- If a subject has one signature feature in every photo (a tattoo, a scar), do not caption it: it becomes
  part of the trigger.
- If a photo is a body shot with no face, still caption it with the trigger; the method needs the whole
  person.
- Do not put quality words or tags ("masterpiece", "4k") in captions.

### 4.5 Captions you write the joined word into yourself

Set `isolation.as_written: true` and write `zwxemwoman smiling in a park` yourself in every caption.
Nothing is rewritten. Give each subject's `trigger` as the joined word (`zwxemwoman`), and, if the word
doesn't end in a class word (woman, man, girl, boy, dog, cat ...), give `what`. `fuse_triggers: false`
instead reads the captions as they are, with the raw trigger kept: the likeness is then bridged to the
class word again, so only use it to compare.

### 4.6 Files without captions

A picture with no `.txt` gets `<trigger> <default_caption>` (subject-level `default_caption` wins over
`dataset.default_caption`; with neither, just the trigger, which reads as `zwxemwoman`). A `prefix.txt`
in the folder (or above it) adds a prefix to every caption below it, `[folder]` standing for the folder's
name. Real captions train better than defaults.

### 4.7 Choosing trigger words

- Made-up words, not names or dictionary words: `zwxem`, `qrxt`, `vytk`.
- Start each subject's trigger with a **different first letter** and share no beginning
  (`zwxem` and `zwxen` split into nearly the same pieces). The run warns when first letters repeat.
- Avoid pieces with a meaning: `qr` summons QR codes; acronyms (`jp`, `fr`, `url`) and short words
  (`elf`, `ox`, `alt`) do the same.
- End in a consonant, so it joins cleanly with the class word (`fyxo`+woman fuses badly).
- Two women's triggers always share the final `woman`, so keep everything before it different.
- Fizgig checked these against Krea 2's text encoder: `dlor`, `gwelt`, `jeskv`, `knurv`, `vorlk`,
  `thrisk`, `zwxem`.

This trainer has no "Check a trigger word" tool; pick from that list, or test your word in ComfyUI with
the text encoder (does it split into meaningful pieces?).

## 5. The config

Start from `templates/examples/train_krea2_lora_isolation.yaml`.

```yaml
isolation:
  subjects:
    - {path: datasets/zwxem, trigger: zwxem}               # what: woman  (only if the captions don't say)
    - {path: datasets/qrxt, trigger: qrxt}
    - {path: datasets/papercut, trigger: papercut, kind: style}
  balance: true
  side_by_side: {enabled: true, strangers_path: C:/data/multi_person_strangers, per_subject: 16, lr_boost: 2.0}
masks: {enabled: false}
train:
  optimizer: adamw            # not automagic3, see 5.2
  learning_rate: 2.0e-4
  ema_decay: 0.98
  dop: true                   # isolation turns it on if you leave it out
  dop_every: 2
  dop_multiplier: 2.0
```

### 5.1 Fizgig's recipe for its results

Rank/alpha 16/16, adaptive learning rate in 2e-4 to 4e-4 (here a constant 2e-4 AdamW), batch 1, EMA
0.98, 0.25 MP, seed 42, checkpoint at epoch 20, DOP every second step at weight 2, 16 side-by-sides per
person at 2x rate. Training time about 1.5x a normal LoRA.

### 5.2 Optimizer note

The side-by-sides' higher rate scales the optimizer's learning rate on their own steps, so it needs an
optimizer with one rate to scale: `adamw`, `adamw8bit` or `muon`. **Automagic3 adapts the rate of every
weight itself, so with it `lr_boost` does nothing**; the log says so and the side-by-sides train at the
normal rate. The rest of isolation works with any optimizer. With a warmup scheduler the boost applies
to the step and the scheduler's rate returns at the next step.

### 5.3 DOP settings under isolation

- `dop_every: 2`, `dop_multiplier: 2.0`: Fizgig's pairing, the same average preservation for about half
  the extra compute of every-step DOP.
- `train.dop_class` is not used for subjects: each subject's own class phrase is. The code only sets a
  placeholder.
- A caption that names no subject gets no DOP pass (it would unlearn the subject). If no caption names
  any subject, the run refuses to start. The log says how many captions have a DOP pass.
- `caption_dropout: 0.05` still applies to the main pass; DOP keeps its own caption.

### 5.4 Length of the run

One epoch = every subject's pictures (repeated by `balance`) plus the side-by-sides, times
`dataset.repeats`. `num_epochs` x that = steps. Example: 36 + 53 pictures, balanced 1:1, plus 32
side-by-sides = 121 pictures per epoch; 20 epochs = 2420 steps.

Fizgig: a woman-and-man pair was right at epoch 10; two women needed about epoch 16, 1.6x: both are held
to the model's own "a woman" and share the joined `woman` piece, so telling them apart takes longer.
Save checkpoints (`save_every`) and compare the in-between epochs.

## 6. Side-by-sides in detail

- 16 pictures per **person** subject (`per_subject`), built before training in
  `<output.path>/side_by_side/` as `maltrainer_sbs_<subject>_<n>.png`, 2:1 frames (`panel` pixels per
  half, default 512), rebuilt every run with the config's seed.
- Each shows one of the subject's photos next to one of the strangers, side chosen at random. Photos
  are the largest centred square, never zoomed. `FaceCrop_*` files (Image Prep) go first; no photo is
  used twice until all have been, and fewer than 16 photos cycle.
- Caption: `<stranger description> on the left, and zwxemwoman on the right, two people side by side, two
  close-up portraits`. The stranger is described first because the text encoder is causal: words after a
  trigger carry it, so the stranger's description stays clean.
- **Needs a strangers folder**: pictures plus a `strangers.json`:

  ```json
  [{"file": "s0.png", "description": "an Indian man in his 30s"}, ...]
  ```

  Fizgig's published set (32 generated people: 8 ethnicities, men and women, 20s to 50s, no real people)
  is `ShootTheSound/fizgig-multi-person-strangers` on Hugging Face (CC BY 4.0) and works as it is. Or
  make your own, with varied people so no one stranger is learned.
- **No face detector.** Fizgig frames photos on the face; here a photo that is not a `FaceCrop_` file is
  centre-cropped. For a full-body photo that leaves little face. Run Image Prep face crops and keep them
  in the subject's folder, or the side-by-sides are weaker.
- `thing` and `style` subjects get none.
- They are normal samples: they get DOP and, with `masks`, no mask (no mask beside them).

## 7. Masks with subjects

`masks.enabled: true` with `floor` (default 0.1) trains each picture where its mask is white, and
optionally `prior` to hold the rest to the base model. See "Masked training" in the Krea 2 guide. For
subjects:

- Masks sit beside each photo in its subject folder (`photo-masklabel.png`).
- For a shared photo the mask must cover **all** the subjects, or the uncovered one trains at the floor.
- Masks help most with outfit and background leakage. Fizgig reports a clearly higher best likeness and
  a faster rise on every recent model.
- A picture without a (usable) mask trains whole, including every side-by-side.

## 8. Reading the log

Healthy start-up lines:

```
Isolation subject zwxem (person, 36 images): captions read as 'zwxemwoman'
Isolation subject qrxt (person, 53 images): captions read as 'qrxtman'
Isolation: 32 side-by-side pictures (trained at x2 on their own steps)
DOP: 121 of 121 captions name a subject (e.g. a woman smiling in a park)
Isolation: subjects repeat [1, 1] times per epoch ([36, 53] images)
```

Warnings to act on:

| Message | Meaning / fix |
|---|---|
| `the captions never say what 'x' is` | Add `what: woman` (or fix the captions) for that subject |
| `triggers ... share a first letter` | Change a trigger |
| `DOP: N caption(s) name no subject` | Those pictures get no DOP pass; check you did not forget a trigger |
| `lr_boost is ignored with Automagic3` | Use `adamw`, `adamw8bit` or `muon` for the boost |
| `no usable strangers.json` | `strangers_path` is wrong or the file is missing |
| `Mask ... isn't the shape of its picture` | The picture was cropped or rotated after the mask was made; remake the mask |

Saved LoRAs record the subjects (`maltrainer_dop`), the balance (`maltrainer_isolation_balance`) and
masks (`maltrainer_masked_training`) in their metadata.

## 9. Prompting the finished LoRA

- **Use the joined word**: `zwxemwoman`, `qrxtman`. Not "zwxem, a woman".
- **Name the other people before your subjects**: `three friends at a garden party with zwxemwoman and
  qrxtman`.
- **Be specific about the others**: `a Japanese man in his 60s with round glasses` keeps its own look
  better than "a man". Describing your subjects helps too (`zwxemwoman with black hair`); with three
  subjects in one picture also give each one's ethnicity.
- **No trigger = the model's own people**: `a woman and a man at a cafe` makes ordinary people.
- Example: `an East Asian man in his 50s in the middle, zwxemwoman on the left and qrxtman on the right, a
  tight close-up photo, cheek to cheek, all looking at the camera`.

How many subjects: 3 or 4 train well together; 2 or 3 of them can share a picture and stay themselves;
two of yours plus a person the model knows almost always works; three of yours plus a known person is hit
and miss.

## 10. A quick check that it worked

Render the same seed four ways, at your best checkpoint:
1. each trigger alone (the subject appears),
2. two triggers in one picture (both faces, distinct),
3. one trigger and a described stranger of the same kind (the stranger stays a stranger: this is the test
   that fails on a normal LoRA),
4. no trigger, "a woman" (the model's own woman).

If 3 or 4 show your subject, train longer with DOP on, add more varied pictures, make sure side-by-sides
were built and boosted (log, optimizer), and check the triggers share no beginning.

## 11. Differences from Fizgig

- No Subjects card or "Check a trigger word" tool (config file instead).
- No face detector for the side-by-sides (face crops first, else a centred square).
- The strangers set is not downloaded for you: point `strangers_path` at it.
- `lr_boost` needs a single-rate optimizer (not Automagic3).
- Class kinds are the same (person, thing, style); there is no separate "correct a trigger word" box:
  use `what`, `kind`.
- Applies to the LoRA trainer only; the Krea 2 finetune does not use it.
