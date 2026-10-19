# MOSS-Image text-encoder tokenizer

The Qwen3.5 tokenizer (vocabulary 248320, shared by every Qwen3.5 size) that MOSS-Image's
Qwen3.5-9B text encoder uses. Vendored so maltrainer can tokenize without a ComfyUI checkout.

Copied from:

- Repository: `comfyanonymous/ComfyUI`
- Directory: `comfy/text_encoders/qwen35_tokenizer`
- Commit: `404d7b99` (2026-03-26, "Support Qwen3.5 text generation models")

Files: `vocab.json`, `merges.txt`, `tokenizer_config.json`, unchanged.

## Notes

- Loaded with `transformers.Qwen2Tokenizer`, the class ComfyUI's `Qwen35Tokenizer` uses, so
  token ids match what ComfyUI feeds the encoder at inference.
- `tokenizer_config.json` carries Qwen3.5's own `pretokenize_regex` and the chat template. The
  template is not used: maltrainer formats the prompt itself, as ComfyUI's
  `MOSSImageTokenizer` does.

To override these at runtime, set `model.tokenizer_path` in the training config.
