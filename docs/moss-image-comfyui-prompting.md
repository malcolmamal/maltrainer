# Using a MOSS-Image finetune in ComfyUI: setup and prompting

How to run a people finetune from [moss-image-finetuning.md](moss-image-finetuning.md) in
ComfyUI, and how to prompt it. The setup below gives very good results in 8 steps: not
necessarily the best possible, but a strong default to start from.

## The setup: 8 steps with the DMD LoRA

| Node | Setting |
|---|---|
| **Load Diffusion Model** | the finetune, e.g. `MOSS/moss_image_ft_people.safetensors` |
| **Load LoRA** (model only) | `Moss/special/MOSS-Image-Base-8-steps-dmd.safetensors`, `strength_model` **1.00** |
| **Load CLIP** | `qwen3.5_9b_int8_convrot.safetensors` (the encoder the finetune trained with) |
| **Load VAE** | `flux2_vae.safetensors` |
| **Empty latent** | 1024 × 1024 |
| **KSampler** | steps **8**, cfg **4.0**, sampler **euler**, scheduler **simple**, denoise **1.00**, seed randomize |

Wiring: Load Diffusion Model → Load LoRA → KSampler `model`; the positive and negative prompts
from CLIP Text Encode into `positive` / `negative`; KSampler → VAE Decode.

- **The DMD LoRA** is lodestones' 8-step distillation LoRA for **MOSS-Image Base** (rank 128).
  The finetune was trained from Base, so the Base version applies; there is an SFT version
  (`MOSS-Image-SFT-8-steps-dmd`) for the SFT model, which is the wrong one here.
- **Shift** needs no node: ComfyUI runs MOSS-Image at shift 5 by default. To change it, add
  **ModelSamplingAuraFlow** (not ModelSamplingSD3, whose timestep scale is 1000x off for MOSS)
  between the LoRA and the KSampler.
- **Resolution:** 1024 × 1024 is what the training previews used. The finetune trained on
  512 px images, but MOSS-Image itself finished pretraining at 1024 px.

### Without the DMD LoRA

The slower, undistilled route: drop the Load LoRA node and use the MOSS template's settings,
res_multistep / simple, 30 steps, CFG 4, with **SkipLayerGuidanceDiTSimple** on layer 22 (start
0, end 0.5) and the negative `worst quality, bad anatomy, watermark, jpeg artifact`.

### The finetune as a LoRA

The finetune can also be loaded as a LoRA on the plain Base model, extracted with
[scripts/extract_lora.py](extract-lora.md) (`moss_image_ft_people_lora_r64/128/256` in
`loras/Moss`): Load Diffusion Model `moss_image_base_bf16` → Load LoRA (the extracted one,
strength 1.0) → Load LoRA (the DMD LoRA) → KSampler with the settings above. Higher ranks keep
more of the finetune; the full checkpoint keeps all of it. That stack is not tested yet: compare
it with the full checkpoint on the same seed before relying on it.

How much a rank keeps, for the 24,000-step people finetune (the size of each layer's change that
the SVD drops, mean over layers): r64 96%, r128 93%, r256 87%. The change is spread over many
directions: at batch size 1 much of it is step-to-step noise (the monitor's drift grew like a
random walk, t^0.45), which no low rank captures and which may not matter for likeness. So a
large dropped share does not by itself mean lost likeness, but it is why the LoRAs need the
side-by-side check.

## Prompting

The finetune learned each person from captions of one shape, so prompts work best in that shape:

```
photo of <trigger>, <a sentence describing the picture>
```

- **`photo of <trigger>`** comes first, exactly as in training: a `prefix.txt` with
  `photo of [folder]` put it in front of every caption, so the trigger is the person's **folder
  name** under `datasets/finetunes/moss/people/` (they are also the keys of `prior.classes` in the
  run's `config.yaml`). One word, lower case, no spaces: `feliciaday`, `keanureeves`.
- **Then one natural sentence**, in the order the captions use: who (`a woman` / `a man`, hair,
  expression), framing, pose, clothing, background, lighting.

```
photo of feliciaday, a woman with reddish-brown wavy hair smiling slightly, close-up with her face
fully visible, looking directly at the camera, wearing a dark top, in front of a white backdrop,
under bright even lighting
```

```
photo of keanureeves, a man with dark hair and a beard, looking slightly to his right, close-up
with his face visible, wearing a brown coat over a blue shirt, against a blurred city background,
under diffused natural light
```

Tips:

- **Name the class word after the trigger** (`a woman`, `a man`), as the captions did. Prior
  preservation tied each trigger to it, and the class word without a trigger still makes a
  generic person, so `photo of a woman, ...` is how you get someone who is not in the finetune.
- **Write "photo of", not "RAW photo of".** The training captions never said RAW, and previews
  prompted with "RAW photo of ..." sometimes painted the word RAW into the background.
- **Use the captions' framing words:** `close-up with face visible`, `head slightly tilted`,
  `looking directly at the camera`. The captions are largely close-ups, which is also what the
  training previews checked; wider shots are less tested, so describe the body and setting
  explicitly there.
- **One person per image.** The dataset has no group shots captioned with two triggers, so two
  people in one prompt tend to blend. That is untested; expect it to need group-shot training data.
- **Negative prompt:** with CFG 4 it has an effect. Start empty (what the training previews used)
  or with `worst quality, bad anatomy, watermark, jpeg artifact`.
