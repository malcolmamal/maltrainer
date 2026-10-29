# Visage-Image text-encoder config

`config.json` is the published Qwen3-0.6B configuration (the values ComfyUI's `Qwen3_06BConfig` also uses), without
the `torch_dtype` entry. It is loaded through `Qwen3Config.from_pretrained` so transformers migrates it to the
installed version's schema (`rope_theta` / `rope_scaling` became a merged `rope_parameters`); the only value
overridden in code is `num_hidden_layers`, cut to the 8 layers Visage-Image conditions on.

The tokenizer is not duplicated: Qwen3 and Qwen3-VL share one vocabulary, so `load_visage_text_encoder` reuses the
set vendored for MiniMax H3 (`models/minimax_h3/assets/text_encoder`). Both it and ComfyUI's `qwen25_tokenizer`
give the same ids for ordinary text. To override it, set `model.tokenizer_path` in the training config.
