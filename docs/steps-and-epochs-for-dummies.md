# Steps, epochs and what happens in one step (for dummies)

A plain-language guide for anyone new to training. It applies to both **LoRA training** and **full finetuning**.

## The flashcard picture

Your training images are a deck of flashcards.

- **Step** = one study session. The model looks at some cards, sees how wrong it was and corrects itself a tiny bit.
- **Epoch** = going through the whole deck once.

```
steps per epoch = samples per epoch / gradient_accumulation_steps
total steps     = epochs x steps per epoch
```

Example: 40 images, `repeats: 1`, `gradient_accumulation_steps: 4` gives 10 steps per epoch. Train for 100 steps and the
model has seen every image 10 times (10 epochs).

### Repeats count too

If a folder has `repeats: 5`, each of its images is dealt five times per epoch. A "sample" in the formula above is
an image times its repeats. 40 images with 5 repeats is 200 samples per epoch.

### Why epochs matter

- **Too few**: the model has not learned your subject yet (weak likeness, generic results).
- **Too many**: it memorises the exact photos instead of the general look (overfitting): same poses, same
  backgrounds, less flexibility with new prompts.

There is no magic number. It depends on the dataset size, the learning rate and what you are teaching. Watch the
previews and stop or pick the checkpoint where the result is good.

## LoRA vs. finetune: same counting, different thing nudged

Steps and epochs are counted exactly the same way. What differs is what gets changed:

| | LoRA | Full finetune |
|---|---|---|
| What changes | a small add-on; the main model is frozen | every weight in the model |
| Memory needed | low | high |
| Output file | small | as big as the whole model |

## What happens during one step

1. **Pick the images.** The trainer takes the next sample(s) and their captions.
2. **Add noise.** Each image is scrambled with random noise, by a random amount, from slightly fuzzy to pure static.
3. **Forward pass.** The model is shown the noisy image and the caption and asked to predict the noise (or the clean
   image, depending on the model).
4. **Loss.** The guess is compared with the truth. Big number = far off, small number = close.
5. **Backward pass.** For every trainable weight, work out which direction would have made the guess a bit better.
   These directions are the *gradients*.
6. **Optimizer update.** Every trainable weight is nudged a small amount in that direction. The size of the nudge is
   the *learning rate*.
7. **Repeat** with the next images.

One step changes the model very slightly. The learning is thousands of tiny nudges adding up.

### The backward pass, explained

The forward pass asks the model for a guess. The backward pass asks: **"whose fault was the mistake?"**

The model is made of millions of tiny dials (the weights). When the guess is wrong, some dials contributed a lot to
the mistake and some barely mattered. The backward pass works backwards from the error, through the model, and for
every trainable dial it records which way to turn it (up or down) and how strongly it mattered. That list is the
gradients. **Nothing changes yet**: it is only the diagnosis. The optimizer update (step 6) is the repair.

A cooking picture: you taste the soup and it is too salty.

- Forward pass = the soup you made. Loss = "too salty".
- Backward pass = working out which ingredients caused it (too much salt, not enough water).
- Optimizer update = adjusting the recipe a little.
- Learning rate = how much you adjust. Change it too much and you ruin the next batch the other way.

In LoRA training only the small add-on has dials to diagnose and turn; the main model is left alone. That is why LoRA
needs less memory.

## Learning rate: how big is each nudge?

The learning rate is the size of each nudge in step 6. It trades speed against safety, like the length of your stride
when walking to a shop.

| | Bigger learning rate | Smaller learning rate |
|---|---|---|
| Speed | learns in fewer steps | needs more steps |
| Risk | overshoots: noisy or "burnt" results, can wreck what the model already knew | steadier and safer |
| Epochs needed | fewer | more |

So a smaller learning rate usually means you need more steps, and so more epochs on the same dataset, to get to the
same place. It is a good rule of thumb, not an exact swap: halving the rate does not always mean exactly doubling the
steps. Some optimizers (such as Prodigy) adjust the rate by themselves, so it matters less there.

## Batch size and gradient accumulation

Looking at several images at once gives a steadier nudge than looking at one. Most trainers do this with a *batch size*.
**maltrainer's image trainers keep `train.batch_size` at 1** and get the same effect with
`gradient_accumulation_steps`:

- Steps 1 to 5 run once per image. Their gradients are **added up**.
- Step 6 (the nudge) runs **once**, after `gradient_accumulation_steps` images.

So with `gradient_accumulation_steps: 4`, one *step* means four images were looked at and the model was nudged once.
It behaves like a batch of 4 and needs the memory of a batch of 1.

Rule of thumb: **one step = one nudge**, and each nudge is based on `gradient_accumulation_steps` images.

## Setting it up in practice

You set the length of a run with `train.max_steps` (or `train.num_epochs` when `max_steps` is 0). To work out the
steps for a number of epochs without loading any model, use `--suggest-steps`:

```bash
python train_moss_image_finetune.py --config config/moss_image_finetune__people.yaml --suggest-steps 15
```

It counts your images, applies repeats and accumulation and prints the suggested `train.max_steps`. See the
[finetuning walkthrough](finetune-tutorial.md) for the full workflow.

## Quick glossary

| Word | Meaning |
|---|---|
| Sample | one image (with its caption) as dealt in an epoch; repeats multiply them |
| Step | one model update (one nudge) |
| Epoch | every sample seen once |
| Loss | how wrong the model's guess was |
| Gradient | the direction that would reduce the loss |
| Learning rate | how big each nudge is |
| Gradient accumulation | add up gradients from several images before one nudge |
| Overfitting | memorising the training photos instead of learning the general look |
