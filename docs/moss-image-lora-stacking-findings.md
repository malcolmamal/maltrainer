# MOSS-Image: two weak LoRAs that stack into a strong one

Findings from the first feliciaday LoRA runs on MOSS-Image Base, October 2026.

**In short:** two separately trained LoRAs were each under-strength on their own. One came from
Automagic3, one from Prodigy. Loaded together, both at strength 1.0, the results were the best so far. The likely
reasons, and what to do about them, follow.

## The runs

Both runs used the same setup:

- 23 images, 512×512
- the bf16 DiT and the int8 Qwen3.5-9B text encoder
- LoRA rank 16, alpha 16, `blocks` targets (252 adapters)
- the report's timestep distribution: logit-normal, mean sigma ~0.74, no shift
- caption dropout 0.05

They differed only in the optimizer.

| | Automagic3 | Prodigy |
|---|---|---|
| Config | `train_moss_image_feliciaday_automagic3.yaml`, run for 2300 steps | `train_moss_image_feliciaday_prodigy.yaml` |
| Optimizer | Automagic3, 8-step sign window, launch 1e-6, EMA 0.98 | Prodigy, lr 1.0, 100 warmup steps, weight decay 0.01, no EMA |
| Checkpoint used | step 500 | step 750 |
| Alone at strength 1.0 | under-strength. It worked at 1.4-1.5 | under-strength. Better at 1.5, but not right |
| Stacked together, both at strength 1.0 | **very good: the best results so far** | |

Observations from the Automagic3 log:

- **Rate.** The adaptive rate peaked at 5.7e-5 around step 200 (MLP 8.4e-5, attention
  2.9e-5). It then decayed steadily to 1.1e-5 at step 2300 (attention 7e-6).
- **Likeness.** Likeness at step 500 already matched step 2300. After about 500 steps the run
  added little.
- **Loss.** Mean loss went 0.578 → 0.546. With these timesteps the loss is dominated by
  high-noise steps and says little about likeness.

## Why stacking works better than one LoRA turned up

Loading two LoRAs adds their weight changes. Each change has two parts:

- **The subject**, the thing both runs set out to learn. The two runs learned it independently,
  but it points the same way in both, so it adds up. Two half-strength LoRAs make one
  full-strength subject.
- **Each run's own errors:** a pose or lighting it latched onto from one photo, drift from its
  particular noise draws, the optimizer's path. These differ between the runs and point in
  different directions, so they partly cancel instead of adding up.

So the stack gets roughly twice the subject with less than twice the errors. One LoRA turned up
to 1.5 scales its subject and its errors together. That fits what was seen: Prodigy at 1.5 was
"better but not exactly right", while the stack looked right.

Averaging or summing separately trained runs like this is known as a "model soup" (for LoRAs,
a "LoRA soup"). It works best when the runs are good but different. Two optimizers are a
strong form of "different".

## What it says about training

Both optimizers came out under-strength at 1.0, so the optimizer is not the main bottleneck.
What the runs share is the more likely cause:

- **The timesteps.** The report's pretraining distribution centres on sigma ~0.74, and only
  ~2 % of steps fall below sigma 0.3. Fine identity detail is mostly learned at mid and low
  noise, so each run spends most of its steps where likeness is decided least.
- **The update size.** Automagic3's rate decayed to a fraction of its peak, and both runs
  stopped well short of strength 1.0. More, larger steps in the right noise range should help.

## What to do with it

1. **Keep the result now.** Merge the pair into one file with `scripts/merge_loras.py` at
   1.0 each, the strengths of the stack (the default, so no `:strength` is needed). The merge
   is exact: one rank-32 LoRA that gives exactly the stacked result at strength 1.0. See
   [merge-loras.md](merge-loras.md).
2. **Test the diagnosis.** Train one run with `timestep_logit_mean: 0` (centred on sigma 0.5)
   and everything else unchanged. If that single LoRA is right at strength 1.0, the timesteps
   were the bottleneck.
3. **Use the soup on purpose.** Even once single runs reach full strength, merging two runs
   should still give a cleaner LoRA than either alone. The runs can differ by optimizer or by
   seed. At ~0.6 s/step, a 1000-step run takes ~10 minutes, so a second run is cheap. When
   merging two full-strength LoRAs, start at about 0.5-0.7 each, not 1.0 + 1.0.

## Open questions

- Was the stack better because of the two optimizers, or would two Prodigy runs with different
  seeds do as well?
- Does the soup still help once single runs are at full strength, or does it mainly make up
  for under-training?
- Which merge strengths work best? Equal strengths, or more weight on one optimizer's LoRA?
