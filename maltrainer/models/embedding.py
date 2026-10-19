"""
Textual Inversion embedding implementation for Z-Image.
"""

from typing import List

import torch
import torch.nn as nn


class TextualInversionEmbedding(nn.Module):
    """Textual inversion embedding that lives in the text encoder's embedding table."""

    def __init__(
        self,
        tokenizer,
        text_encoder: nn.Module,
        trigger_token: str = "sks",
        num_vectors: int = 1,
        init_text: str = "a photo",
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.trigger_token = trigger_token
        self.num_vectors = num_vectors
        self.init_text = init_text

        # Add tokens to tokenizer
        self.token_ids = self._add_tokens()

        # Resize text encoder embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Initialize with init_text embeddings
        self._initialize_embeddings()

    def _add_tokens(self) -> List[int]:
        """Add placeholder tokens to the tokenizer."""
        tokens = [self.trigger_token]

        # Add additional tokens for multi-vector embeddings
        if self.num_vectors > 1:
            tokens.extend([f"{self.trigger_token}_{i}" for i in range(1, self.num_vectors)])

        # Add to tokenizer
        num_added = self.tokenizer.add_tokens(tokens)

        if num_added != self.num_vectors:
            raise ValueError(
                f"Expected to add {self.num_vectors} tokens, but added {num_added}. "
                f"Token '{self.trigger_token}' may already exist."
            )

        # Get token IDs
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        return token_ids

    def _initialize_embeddings(self):
        """Initialize new embeddings with init_text embeddings."""
        # Encode init text to get initial embeddings
        init_token_ids = self.tokenizer.encode(self.init_text, add_special_tokens=False)

        # Get embedding layer
        embedding_layer = self.text_encoder.get_input_embeddings()

        with torch.no_grad():
            for i, token_id in enumerate(self.token_ids):
                # Use init token or repeat last one
                init_idx = init_token_ids[i] if i < len(init_token_ids) else init_token_ids[-1]
                embedding_layer.weight[token_id] = embedding_layer.weight[init_idx].clone()

    def get_embeddings(self) -> torch.Tensor:
        """Get the learned embedding vectors."""
        embedding_layer = self.text_encoder.get_input_embeddings()
        embeddings = torch.stack([embedding_layer.weight[tid] for tid in self.token_ids])
        return embeddings

    def set_embeddings(self, embeddings: torch.Tensor):
        """Set the embedding vectors."""
        embedding_layer = self.text_encoder.get_input_embeddings()
        with torch.no_grad():
            for i, token_id in enumerate(self.token_ids):
                embedding_layer.weight[token_id] = embeddings[i]

    def get_state_dict(self) -> dict:
        """Get state dict for saving (compatible with SD format)."""
        embeddings = self.get_embeddings()

        return {
            "embedding": embeddings,
        }

    def load_state_dict(self, state_dict: dict):
        """Load embeddings from state dict."""
        embeddings = state_dict["embedding"]
        self.set_embeddings(embeddings)

    def parameters(self):
        """Return only the embedding parameters."""
        embedding_layer = self.text_encoder.get_input_embeddings()
        return [embedding_layer.weight[tid] for tid in self.token_ids]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward is handled by text encoder, this is just for optimizer."""
        pass
