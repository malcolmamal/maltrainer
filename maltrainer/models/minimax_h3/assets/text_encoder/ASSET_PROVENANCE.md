# MiniMax H3 text-encoder metadata

Configuration, tokenizer, and processor files for the Qwen3-VL text encoder that
MiniMax H3 conditions on. Vendored here so maltrainer can build the tokenizer and
config without depending on a sibling trainer checkout being present.

Copied from:

- Repository: `MiniMaxAI/MiniMax-H3`
- Subdirectory: `FL2VA/text_encoder`
- Revision: `fa9c8ab1eaa21c8ae25e7e40b83b2e6002f340af`

Only the files needed to construct the H3 Qwen3-VL configuration and processor are
included. Model weights and their index, the unused chat template, and the redundant
slow-tokenizer vocabulary files are intentionally excluded.

## Notes

- `tokenizer.json` / `tokenizer_config.json` back `AutoTokenizer`. H3 tokenizes raw
  prompt text with `add_special_tokens=False` and no chat template, so the tokenizer's
  own template is irrelevant here.
- `config.json` is the single source of truth for the model config, loaded through
  `Qwen3VLConfig.from_pretrained` so transformers migrates it to the installed version's
  schema. The only value overridden in code is `text_config.num_hidden_layers`, truncated
  to the 50 layers H3 conditions on (this file declares the model's full 64). Do not
  re-transcribe these values into Python: field names move between transformers releases
  (`rope_theta`/`rope_scaling` became a merged `rope_parameters`), and `from_pretrained`
  handles that translation while hand-written kwargs silently do not.
- `preprocessor_config.json` / `video_preprocessor_config.json` are unused by training
  (prompts are text-only) and kept only so the vendored set stays a faithful subset.

To override these at runtime, set `model.tokenizer_path` in the training config.
