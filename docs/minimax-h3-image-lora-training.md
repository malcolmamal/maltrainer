# MiniMax H3 Image LoRA training

`train_minimax_h3_image_lora.py` trains LoRAs for MiniMax H3 Image (3 November 2026), MiniMax's picture model, from
ComfyUI's single-file checkpoints. It is **not** [MiniMax H3](minimax-h3-lora-training.md), the audio and video model:
the layers are H3's, but the weights moved a long way in H3 Image's training, so LoRAs and training adapters made for
H3 do not carry over (and the other way round).

```bash
python train_minimax_h3_image_lora.py --config templates/examples/train_minimax_h3_image_lora.yaml
```

The defaults are Fizgig's Standard preset: rank 8, Automagic3 from 2.8e-4, about 0.5 MP, EMA 0.98, 30 epochs,
Fizgig's training adapter and likeness optimization.

## Model files

| Key | File |
|---|---|
| `model.transformer_path` | `minimax_h3_image_pruned_int8_convrot.safetensors` (Comfy-Org/MiniMax-H3-Image) |
| `model.vae_path` | `minimax_h3_image_vae_fp16.safetensors` (its encoder is H3's; the decoder is not used here) |
| `model.text_encoder_path` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, the same file as H3's |
| `model.training_adapter_path` | `fizgig_minimax_h3_image_training_adapter.safetensors` (ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter, 0.15 GB) |

Not the Circlestone adapter (`circlestone-labs/MiniMax-H3-Image-Training-Adapter`): despite its name it is for the H3
video model.

## How it trains

- **The checkpoint's own noise levels.** H3 Image is distilled to 6 fixed levels (1.0, 0.968, 0.888, 0.750, 0.533,
  0.232), each with its own output head. A step draws one level and runs the model through that level's head, the
  exact inputs it sees when it renders. The head for level k is `rows[0] + rows[k]` of `final_layer.video_out`
  (`rows[0]` for level 0), as in ComfyUI. The grid is read from the file's metadata.
- **Loss.** Plain flow matching on `x0 - noise` (H3's convention), `x_t = (1 - sigma) * x0 + sigma * noise`.
- **`train.likeness_opt`** (on in the example): 75 % of steps at the level below sigma 0.5 (the nearly finished picture,
  where skin, eyes and fine detail are made) for epochs 1 to 8, then 50 % and 75 % in turn. `false` draws the six
  levels evenly (Fizgig's Style, Edit and Slider presets).
- **Training adapter:** frozen at `training_adapter_multiplier` 1.0 for every step, never saved in the LoRA.
- **No guidance loss, no audio, stills only.** The model is distilled without CFG; clips and recordings in the dataset
  folder are ignored. Batch size is 1; use `gradient_accumulation_steps`.
- **Latents** are the posterior mean from a whole-image encode. ComfyUI draws a seeded sample and tiles at 256 px; the
  difference is small and not measured here.

## Not done yet

Previews (they need the image VAE's ViT decoder and the 6-step sampler), edit LoRAs with reference pictures, slider
LoRAs and masked training. Not yet run on the real 21 GB checkpoint; the tests run the trainer on tiny CPU models.

## Saved files

`minimax_h3_image_lora_step_final.safetensors` in `output.path`, ComfyUI-named (`diffusion_model.blocks.N...`), so
ComfyUI's standard Load LoRA takes it. Check it in ComfyUI with `--disable-dynamic-vram` (see Fizgig's
`docs/MINIMAX_IMAGE.md`).
