"""INT8 ConvRot quantized embedding table of the Qwen3-VL text encoder."""

from __future__ import annotations

import torch
from torch import nn

from maltrainer.models.minimax_h3.int8_convrot import rotate_activation


class Int8Embedding(nn.Module):
    """INT8 ConvRot quantized embedding module.

    The checkpoint stores the table already Hadamard-rotated -- its ``comfy_quant`` marker
    carries ``convrot``/``convrot_groupsize`` just like the ConvRot linears, whose forward
    rotates the *activation* to match the rotated weight. An embedding lookup has no
    activation to rotate, so the rotation has to be undone here instead; the Hadamard is an
    involution, so the same ``rotate_activation`` call inverts it.

    Skipping this does not fail loudly. The model still runs and still emits
    plausible-looking hidden states -- they just carry no prompt, which downstream looks
    like a diffusion LoRA that trains to a low loss and has no subject likeness.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, group_size: int = 0):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), dtype=torch.int8), requires_grad=False)
        self.register_buffer("scale_weight", torch.ones((num_embeddings, 1), dtype=torch.float32))
        self.register_buffer("int8_convrot_groupsize", torch.tensor(int(group_size), dtype=torch.int32))
        self.out_dtype = torch.bfloat16

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        w = torch.embedding(self.weight, input_ids)
        scale = torch.embedding(self.get_buffer("scale_weight"), input_ids)
        embeds = w.float() * scale
        group_size = int(self.get_buffer("int8_convrot_groupsize").item())
        if group_size:
            embeds = rotate_activation(embeds, group_size)
        return embeds.to(self.out_dtype)
