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
- `config.json` supplies the vision-tower config. The **text** side is not read from it:
  `text_encoder.py::_minimax_h3_text_config` pins those values in code and truncates to
  the 50 layers H3 uses (this file declares the full 64).
- `preprocessor_config.json` / `video_preprocessor_config.json` are unused by training
  (prompts are text-only) and kept only so the vendored set stays a faithful subset.

To override these at runtime, set `model.tokenizer_path` in the training config.
