# MiniMax H3 LoRA training

`train_minimax_h3_lora.py` trains LoRAs for MiniMax H3, MiniMax's joint audio+video model, from
ComfyUI's single-file checkpoints. You don't need diffusers folders. One dataset folder can mix
stills, video clips with sound, and audio-only recordings.

```bash
python train_minimax_h3_lora.py --config templates/examples/train_minimax_h3_lora_fizgig_recipe.yaml
```

| Config | Use it for |
|---|---|
| [`train_minimax_h3_lora_fizgig_recipe.yaml`](../templates/examples/train_minimax_h3_lora_fizgig_recipe.yaml) | Character/style LoRAs from photos, with Fizgig's default recipe (see below) |
| [`train_minimax_h3_lora_audio_video.yaml`](../templates/examples/train_minimax_h3_lora_audio_video.yaml) | Photos, clips with sound and voice recordings in one run |
| [`train_minimax_h3_lora_tuned.yaml`](../templates/examples/train_minimax_h3_lora_tuned.yaml) | The earlier Prodigy + contrastive-guidance recipe |

## Model files

All files are ComfyUI's releases.

| Key | File | Needed for |
|---|---|---|
| `model.transformer_path` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | Always |
| `model.vae_path` | `minimax_h3_video_vae_fp16.safetensors` | Stills and clips |
| `model.audio_vae_path` | `minimax_h3_audio_vae_fp32.safetensors` | Clip soundtracks and audio files |
| `model.text_encoder_path` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | Always |
| `model.training_adapter_path` | A de-distillation adapter (see below) | Recommended |
| `model.training_adapter_adaln_grid` | Fizgig's `src/fizgig/assets/h3_silu_temb_grid.safetensors` | Only for adapters with full-model AdaLN rows |

The base runs on its own int8 ConvRot weights, so it matches ComfyUI's inference. It is never
re-quantized.

The original release's audio VAE (`FL2VA/audio_vae/model.safetensors`, weight-norm format) also
loads.

## Dataset

Put everything in one folder. Each file can have a caption in a `<stem>.txt` file next to it. A
file without one falls back to `dataset.trigger` / `dataset.default_caption`, as before.

| Files | Kind | What trains |
|---|---|---|
| `.jpg .jpeg .png .webp` | still | Video (1 latent frame) |
| `.mp4 .mov .mkv .webm .avi .m4v` | clip | Video and its soundtrack |
| `.wav .mp3 .flac .m4a .ogg .opus .aac` | audio | Audio only |

**Clips**
- Clips are resampled to H3's 24 fps by timestamp, so 30 or 60 fps footage keeps real-time
  speed.
- A clip trains video only when it has no audio track, when its track is digital silence, or
  when its file name stem ends in `_mute`.

**Lengths**

Clip and audio lengths snap down to H3's frame grid (17k + 5 frames at 24 fps):

| Frames | 22 | 39 | 56 | 73 | 90 | 107 | 124 |
|---|---|---|---|---|---|---|---|
| Seconds | 0.92 | 1.63 | 2.33 | 3.04 | 3.75 | 4.46 | 5.17 |

- `dataset.max_frames` caps the window.
- With `long_media: first`, each file contributes its first window.
- With `long_media: split`, each file is cut into consecutive windows. Every window shares the
  file's caption, so don't split a file whose caption is a verbatim transcript.

**Audio items**
- An audio item gets a zeros video placeholder (`audio_placeholder_resolution`, 128 px by
  default). Its video loss is dropped. Fizgig handles voice items the same way.
- Recordings shorter than 22 frames (0.92 s) are skipped.

**Captions for sound**

Describe the sound in the caption: speech verbatim, who is speaking, the ambience, the music.
The text encoder is the only place where the model learns what a clip should sound like.

**Resizing**
- Stills and clips keep their aspect ratio (`aspect_mode: bucket`). The area is about
  `resolution`² for stills and `clip_resolution`² for clips, and each side is a multiple of 32.
- `aspect_mode: stretch` keeps the trainer's old square resize.

**Dependencies**

Decoding needs PyAV (`pip install av`, listed in `requirements.txt`), which bundles FFmpeg. A
dataset with only stills doesn't import it.

## How the audio stream is trained

H3 packs one sequence per sample: `[text | audio | video]`. ComfyUI never runs the model without
audio rows. maltrainer packs them for every kind of sample, so the frozen base always sees its
inference layout:

| Kind | Audio rows | Loss |
|---|---|---|
| Still | 2 latent frames x 2 channels of silence (x0 = 0), noised at the audio sigma | Video only |
| Clip with sound | The soundtrack, noised at the audio sigma | Video + `audio_loss_weight` x audio |
| Audio file | The recording, noised at the audio sigma | Audio only |

**Two schedules**

- The streams run on different flow shifts: video 12, audio 3. The audio sigma is always the
  closed-form image of the video sigma (video 0.5 gives audio 0.2), because those are the only
  pairs the model meets at inference. This holds whatever `timestep_distribution` you pick for
  the video sigma.
- Each stream gets its own AdaLN timestep row.

**Audio geometry**

- Audio latents run at 40 per second, with stereo packed channel-major.
- Audio rows share the media clock's origin with the video. Their w position is pinned to the
  video grid's first column (left channel) and last column (right channel).
- Audio windows are cut to exactly `round(frames / 24 x 40) x 800` samples, so the encoder
  produces the row count the layout expects. The trainer checks this rather than trusting it.

**Contrastive guidance**

When `do_guidance_loss` is on, contrastive guidance extrapolates the audio target as well as the
video one. Each stream decays on its own sigma.

`train.pack_silent_audio: false` brings back the old audio-free layout for stills, for A/B
comparisons. `train.train_audio: false` ignores soundtracks and audio files.

## The Fizgig recipe

Fizgig produces strong H3 LoRAs on near-default settings. Its "Fast" preset with "Likeness and
Style" uses the settings below, and each one is available here:

| Setting | Fizgig | maltrainer key |
|---|---|---|
| De-distillation adapter, frozen, strength 1.0, never saved | Circlestone (default) or Ostris v1 | `model.training_adapter_path`, `lora.training_adapter_multiplier` |
| LoRA only on blocks 20–49; refiner and AdaLN frozen | yes | `lora.block_range: [20, 49]`, `include_token_refiner: false`, `include_adaln: false` |
| 60 % of steps below sigma 0.5 (uniform u, shift 0.667) | "Likeness and Style" | `timestep_distribution: lownoise`, `lownoise_pct: 60` |
| Automagic3 starting at 1e-6, no weight decay | yes | `optimizer: automagic3`, `learning_rate: 1.0e-6`, `weight_decay: 0` |
| EMA 0.98 with warm-up; checkpoints save the average | yes | `train.ema_decay: 0.98` |
| Caption dropout 0.05 (empty-prompt embedding) | yes | `train.caption_dropout: 0.05` |
| Rank 8 / alpha 8, about 0.25 MP aspect buckets, batch 1 | yes | `lora.rank`, `dataset.resolution: 512` |
| Silent audio rows packed with their own timestep | yes | default (`pack_silent_audio: true`) |

**Training adapters**

- Ostris fl2va v1: <https://huggingface.co/ostris/minimax_h3_training_adapter>. This was
  Fizgig's only adapter up to v6.2.0, and the recipe configs use it. Fizgig now calls it the
  best choice for video. Don't use the `ref2va` file from the same repo: it belongs to the
  ref2va base, not the fl2va checkpoint trained here.
- Circlestone image adapter: <https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter>.
  This has been Fizgig's default since v6.3.0 (23 Sep 2026). In Fizgig's A/Bs it trained
  sharper photo LoRAs that follow prompts better.

**Full-model AdaLN rows**

- An adapter trained on the full (unpruned) model can carry AdaLN rows that read the
  2688-wide time embedding. The pruned checkpoint doesn't compute that embedding.
- Set `model.training_adapter_adaln_grid` and those rows run through the precomputed grid, as
  Fizgig does.
- Without the grid they are skipped, and the log says so.
- The loader accepts both PEFT and Kohya key names.

**Differences from Fizgig**

- The adapters train in fp32. Fizgig uses bf16 with stochastic rounding.
- `max_grad_norm` clipping is active. In Fizgig it is effectively a no-op under Automagic3.
- TREAD token routing isn't implemented. Fizgig uses it only for clips.

## Configuration reference (new keys)

| Key | Default | Meaning |
|---|---|---|
| `dataset.media` | `[image, clip, audio]` | Kinds of file to train on |
| `dataset.aspect_mode` | `bucket` | `bucket` (keep aspect) or `stretch` (old square resize) |
| `dataset.clip_resolution` | `resolution` | Area target for clip frames |
| `dataset.max_frames` | `73` | Clip/audio window, snapped down to 17k + 5 |
| `dataset.long_media` | `first` | `first` or `split` |
| `dataset.max_segments_per_file` | `16` | Cap for `split` |
| `dataset.clip_audio` | `true` | Train clip soundtracks |
| `dataset.audio_placeholder_resolution` | `128` | Size of the zeros video for audio items |
| `lora.block_range` | all | Inclusive `[first, last]` DiT blocks with adapters |
| `lora.include_token_refiner` | `true` | Adapters on the 2 text-refiner blocks |
| `train.train_audio` | `true` | Use soundtracks and audio files |
| `train.audio_loss_weight` | `1.0` | Weight of the audio MSE (audio is about 4 % of a clip's tokens) |
| `train.pack_silent_audio` | `true` | Pack silence audio rows for stills and muted clips |
| `train.timestep_distribution: lownoise` + `lownoise_pct` | `60` | Uniform u through the shift that puts that share of steps below sigma 0.5 |
| `train.ema_decay` / `train.ema_warmup` | off / `true` | EMA of the adapters |
| `train.caption_dropout` | `0` | Probability of training a step on the empty prompt |
| `model.audio_vae_path` | — | Audio VAE, required when audio is trained |
| `model.training_adapter_adaln_grid` | — | See above |

`train.batch_size` must be 1. Every sample packs its own sequence, and H3's timestep modulation
is shared across a batch. Use `gradient_accumulation_steps` for a larger effective batch.

## Verified equivalence

These checks ran on CPU with random weights:

- maltrainer's forward over the `[text | audio | video]` layout matches ComfyUI's
  `MiniMaxH3Model` to about 5e-7 relative. That holds for both the video and audio outputs, for
  a still and a 22-frame clip, at high and low noise. The old audio-free layout differed by
  about 10 % on the same still.
- The audio encoder matches ComfyUI's `MiniMaxH3AudioVAE.encode` exactly.
- The packed position grid matches ComfyUI's `PackedLayout` exactly.
- Clip encoding matches ComfyUI's chunking: each 17 frames encode independently and the trailing
  tokens are dropped, so 22 frames give 7 latent frames.

These checks say nothing about LoRA quality. That needs real training runs.
