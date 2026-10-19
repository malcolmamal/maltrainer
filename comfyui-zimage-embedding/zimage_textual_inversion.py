"""
ComfyUI custom node for Z-Image Textual Inversion support.

This node provides text encoding with textual inversion embeddings for Z-Image models.
"""

from pathlib import Path

import safetensors.torch as safetensors
import torch
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM


class ZImageTextEncodeWithEmbeddings:
    """
    ComfyUI node for encoding text with Z-Image textual inversion embeddings.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "a photo of <trigger>"}),
                "model_id": ("STRING", {"default": "Tongyi-MAI/Z-Image"}),
            },
            "optional": {
                "embedding_file": ("STRING", {"default": ""}),
                "token_string": ("STRING", {"default": "<*>"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def __init__(self):
        self.tokenizer = None
        self.text_encoder = None
        self.model_id = None

    def load_model(self, model_id):
        """Load Qwen components for Z-Image if not already loaded."""
        if self.model_id != model_id or self.tokenizer is None:
            self.model_id = model_id
            self.tokenizer = Qwen2Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
            self.text_encoder = Qwen3ForCausalLM.from_pretrained(
                model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
            )
            self.text_encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return self.tokenizer, self.text_encoder

    def load_embedding(self, embedding_path, token_string):
        """Load textual inversion embedding and inject into text encoder."""
        if not embedding_path or not Path(embedding_path).exists():
            return

        # Load the embedding
        embedding_data = safetensors.load_file(embedding_path)

        # Find embedding vectors
        embedding_vectors = None

        # Priority 1: Check for 'emb_params' (common format)
        if "emb_params" in embedding_data:
            embedding_vectors = embedding_data["emb_params"]
        # Priority 2: Check for 'embedding' (our trainer format)
        elif "embedding" in embedding_data:
            embedding_vectors = embedding_data["embedding"]
        # Priority 3: Heuristic search for correct shape
        else:
            for key, tensor in embedding_data.items():
                if tensor.dim() == 2 and tensor.shape[0] <= 16:  # num_vectors x dim
                    embedding_vectors = tensor
                    break

        if embedding_vectors is None:
            print(f"Warning: Could not find embedding vectors in {embedding_path}")
            return

        # Add token if not present
        if token_string not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([token_string])
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Get token id
        token_id = self.tokenizer.convert_tokens_to_ids(token_string)

        # Set the embedding
        with torch.no_grad():
            self.text_encoder.get_input_embeddings().weight[token_id : token_id + embedding_vectors.shape[0]] = (
                embedding_vectors.to(self.text_encoder.device, self.text_encoder.dtype)
            )

        print(f"Loaded {embedding_vectors.shape[0]} embedding vectors for token '{token_string}'")

    def encode(self, text, model_id, embedding_file="", token_string="<*>"):
        """Encode text with optional embedding injection."""
        # Load model components
        tokenizer, text_encoder = self.load_model(model_id)

        # Load embedding if provided
        if embedding_file:
            self.load_embedding(embedding_file, token_string)

        # Tokenize text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(text_encoder.device) for k, v in inputs.items()}

        # Encode text
        with torch.no_grad():
            text_embeds = text_encoder(**inputs).last_hidden_state

        # Return in ComfyUI conditioning format
        # For Z-Image, we return the text embeddings directly
        return ([[text_embeds, {"context": text_embeds}]],)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ZImageTextEncodeWithEmbeddings": ZImageTextEncodeWithEmbeddings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTextEncodeWithEmbeddings": "Z-Image Text Encode (with Embeddings)",
}
