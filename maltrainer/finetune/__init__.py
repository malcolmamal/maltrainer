"""
Model-agnostic pieces of a full finetune.

- ``selection``: which parameters train (name prefixes, Linear weights only)
- ``classes``: triggers and their class words, for class captions
- ``prior``: prior preservation (DOP) against the original model's own predictions
- ``regularisation``: class images mixed in at a reduced step size
- ``extract``: a LoRA from the difference between a finetuned model and its base

Each model brings only what is its own: its default trainable set, its forward and caption
encoding, and how its checkpoint file stores weights (e.g. a fused qkv). MOSS-Image's are in
maltrainer/models/moss_image/finetune.py.
"""
