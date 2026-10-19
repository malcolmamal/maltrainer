"""
Runtime support for Comfy NVFP4/AWQ quantized text encoder checkpoints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

logger = logging.getLogger(__name__)

COMFY_QUANT_SUFFIX = ".comfy_quant"
NVFP4_BLOCK_SIZE = 16
_DEQUANT_CHUNK_ELEMENTS = 16 * 1024 * 1024


def has_comfy_quantized_layers(path: str | Path) -> bool:
    """Return whether a safetensors checkpoint contains Comfy quantization markers."""
    path = Path(path)
    if not path.is_file():
        return False
    with safe_open(path, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        return any(key.endswith(COMFY_QUANT_SUFFIX) for key in source_keys)


def _parse_quant_config(blob: torch.Tensor) -> dict:
    try:
        value = json.loads(bytes(blob.cpu().tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Comfy quantization marker") from exc
    if not isinstance(value, dict):
        raise TypeError("Comfy quantization marker must contain a JSON object")
    return value


def unswizzle_nvfp4_scales(scales: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """Convert the cuBLAS 128x4 blocked scale layout to row-major order."""
    row_blocks = (rows + 127) // 128
    column_blocks = (columns + 3) // 4
    padded_rows = row_blocks * 128
    padded_columns = column_blocks * 4
    expected_elements = padded_rows * padded_columns
    if scales.numel() != expected_elements:
        raise ValueError(
            f"Invalid NVFP4 scale tensor: expected {expected_elements} elements for {rows}x{columns}, got {scales.numel()}"
        )
    values = scales.reshape(-1, 32, 16)
    values = values.reshape(-1, 32, 4, 4).transpose(1, 2)
    values = values.reshape(row_blocks, column_blocks, 4, 32, 4)
    values = values.reshape(row_blocks, column_blocks, 128, 4)
    values = values.permute(0, 2, 1, 3).reshape(padded_rows, padded_columns)
    return values[:rows, :columns].contiguous()


_NVFP4_LUT_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

def get_nvfp4_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    if key in _NVFP4_LUT_CACHE:
        return _NVFP4_LUT_CACHE[key]
    e2m1 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=device, dtype=torch.float32)
    bytes_u8 = torch.arange(256, device=device, dtype=torch.uint8)
    high = bytes_u8 >> 4
    low = bytes_u8 & 15
    high_val = e2m1[(high & 7).long()] * torch.where((high & 8) > 0, -1.0, 1.0)
    low_val = e2m1[(low & 7).long()] * torch.where((low & 8) > 0, -1.0, 1.0)
    lut = torch.stack((high_val, low_val), dim=-1).to(dtype=dtype)
    _NVFP4_LUT_CACHE[key] = lut
    return lut


class ComfyInt8Embedding(nn.Module):
    """Per-row INT8 embedding used by the quantized H3 Qwen3-VL checkpoint."""

    def __init__(self, qweight: torch.Tensor, scales: torch.Tensor, output_dtype: torch.dtype) -> None:
        super().__init__()
        if qweight.ndim != 2 or qweight.dtype is not torch.int8:
            raise ValueError("Comfy INT8 embedding weight must be a rank-2 int8 tensor")
        if scales.numel() != qweight.shape[0]:
            raise ValueError("Comfy INT8 embedding must have one scale per row")
        self.num_embeddings, self.embedding_dim = qweight.shape
        self.output_dtype = output_dtype
        self.register_buffer("qweight", qweight.contiguous(), persistent=False)
        self.register_buffer(
            "scales_u8",
            scales.detach().float().reshape(-1).contiguous().view(torch.uint8),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        lookup_ids = input_ids.reshape(-1).to(self.qweight.device)
        rows = self.qweight.index_select(0, lookup_ids).float()
        scales = self.scales_u8.view(torch.float32).index_select(0, lookup_ids)
        output = (rows * scales.unsqueeze(1)).to(self.output_dtype)
        return output.to(input_ids.device).reshape(*input_ids.shape, self.embedding_dim)


class ComfyNvfp4Linear(nn.Module):
    """NVFP4/AWQ Linear that keeps packed weights and dequantizes each forward."""

    def __init__(
        self,
        source: nn.Linear,
        packed_weight: torch.Tensor,
        blocked_scales: torch.Tensor,
        per_tensor_scale: torch.Tensor,
        pre_quant_scale: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.output_dtype = output_dtype
        if packed_weight.dtype is not torch.uint8 or packed_weight.shape != (
            self.out_features,
            self.in_features // 2,
        ):
            raise ValueError(
                f"Invalid NVFP4 packed weight shape {tuple(packed_weight.shape)} for "
                f"Linear({self.in_features}, {self.out_features})"
            )
        if self.in_features % NVFP4_BLOCK_SIZE != 0:
            raise ValueError("NVFP4 Linear input size must be divisible by 16")
        if per_tensor_scale.numel() != 1:
            raise ValueError("NVFP4 weight_scale_2 must be a scalar")
        if pre_quant_scale is not None and pre_quant_scale.numel() != self.in_features:
            raise ValueError("NVFP4 pre_quant_scale must have one value per input column")

        scales = unswizzle_nvfp4_scales(
            blocked_scales.view(torch.float8_e4m3fn),
            self.out_features,
            self.in_features // NVFP4_BLOCK_SIZE,
        )
        self.register_buffer("packed_weight", packed_weight.contiguous(), persistent=False)
        self.register_buffer("scales_u8", scales.view(torch.uint8), persistent=False)
        self.register_buffer(
            "per_tensor_scale_u8",
            per_tensor_scale.detach().float().reshape(1).contiguous().view(torch.uint8),
            persistent=False,
        )
        if pre_quant_scale is not None:
            if pre_quant_scale.dtype is not torch.bfloat16:
                raise ValueError(f"NVFP4 pre_quant_scale must be bfloat16, got {pre_quant_scale.dtype}")
            self.register_buffer(
                "pre_quant_scale_u8",
                pre_quant_scale.detach().reshape(-1).contiguous().view(torch.uint8),
                persistent=False,
            )
        self.register_parameter("bias", source.bias)

    def _pre_quant_scale(self) -> torch.Tensor | None:
        value = getattr(self, "pre_quant_scale_u8", None)
        return None if value is None else value.view(torch.bfloat16)

    @torch.no_grad()
    def dequantize_weight(self, dtype: torch.dtype) -> torch.Tensor:
        lut = get_nvfp4_lut(self.packed_weight.device, dtype)
        decoded = lut[self.packed_weight.long()].reshape(self.out_features, self.in_features)
        
        block_scales = self.scales_u8.view(torch.float8_e4m3fn).to(dtype=dtype)
        per_tensor = self.per_tensor_scale_u8.view(torch.float32).reshape(())
        block_scales = block_scales * per_tensor
        
        decoded = decoded.view(self.out_features, self.in_features // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
        decoded = decoded * block_scales.unsqueeze(-1)
        return decoded.reshape(self.out_features, self.in_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.packed_weight.device != inputs.device:
            self.to(inputs.device)
        pre_quant_scale = self._pre_quant_scale()
        if pre_quant_scale is not None:
            inputs = inputs * pre_quant_scale.to(dtype=inputs.dtype)
        weight = self.dequantize_weight(inputs.dtype)
        return F.linear(inputs, weight, self.bias)


def _replace_submodule(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, attribute = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, attribute, replacement)


@torch.no_grad()
def load_comfy_quantized_state_dict(
    model: nn.Module,
    checkpoint: Path | str,
    *,
    key_map: Callable[[str], str],
    output_dtype: torch.dtype,
) -> int:
    """Load a Comfy NVFP4/INT8 safetensors checkpoint into a model."""
    checkpoint = Path(checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        marker_keys = [key for key in source_keys if key.endswith(COMFY_QUANT_SUFFIX)]
        if not marker_keys:
            raise ValueError(f"{checkpoint.name} does not contain Comfy quantized layers")

        consumed: set[str] = set()
        format_counts: dict[str, int] = {}
        for marker_key in marker_keys:
            prefix = marker_key[: -len(COMFY_QUANT_SUFFIX)]
            config = _parse_quant_config(handle.get_tensor(marker_key))
            quant_format = config.get("format")
            if not isinstance(quant_format, str):
                raise TypeError(f"{prefix} has no quantization format")
            format_counts[quant_format] = format_counts.get(quant_format, 0) + 1
            module_path = key_map(prefix)
            module = model.get_submodule(module_path)
            weight_key = f"{prefix}.weight"
            scale_key = f"{prefix}.weight_scale"
            if weight_key not in source_keys or scale_key not in source_keys:
                raise ValueError(f"{prefix} is missing its quantized weight or scale")
            weight = handle.get_tensor(weight_key)
            weight_scale = handle.get_tensor(scale_key)

            consumed_keys = {marker_key, weight_key, scale_key}
            if isinstance(module, nn.Embedding):
                if quant_format != "int8_tensorwise":
                    raise ValueError(f"Unsupported Comfy embedding quantization {quant_format!r} on {prefix}")
                replacement = ComfyInt8Embedding(weight, weight_scale, output_dtype)
            elif isinstance(module, nn.Linear):
                if quant_format != "nvfp4":
                    raise ValueError(f"Unsupported Comfy Linear quantization {quant_format!r} on {prefix}")
                scale_2_key = f"{prefix}.weight_scale_2"
                if scale_2_key not in source_keys:
                    raise ValueError(f"{prefix} is missing weight_scale_2")
                pre_scale_key = f"{prefix}.pre_quant_scale"
                replacement = ComfyNvfp4Linear(
                    module,
                    weight,
                    weight_scale,
                    handle.get_tensor(scale_2_key),
                    handle.get_tensor(pre_scale_key) if pre_scale_key in source_keys else None,
                    output_dtype,
                )
                consumed_keys.add(scale_2_key)
                if pre_scale_key in source_keys:
                    consumed_keys.add(pre_scale_key)
            else:
                raise TypeError(f"Comfy quant marker {prefix} maps to {type(module).__name__}; expected Linear or Embedding")
            _replace_submodule(model, module_path, replacement)
            consumed.update(consumed_keys)

        expected = set(model.state_dict())
        state_dict: dict[str, torch.Tensor] = {}
        for source_key in source_keys:
            if source_key in consumed:
                continue
            source_prefix, separator, suffix = source_key.rpartition(".")
            if not separator:
                raise ValueError(f"{checkpoint.name} contains unexpected key {source_key!r}")
            target_key = f"{key_map(source_prefix)}.{suffix}"
            if target_key not in expected:
                raise ValueError(f"{checkpoint.name} contains unexpected key {source_key!r}")
            state_dict[target_key] = handle.get_tensor(source_key)

    missing = sorted(expected - set(state_dict))
    if missing:
        raise ValueError(f"{checkpoint.name} is missing {len(missing)} tensor(s), examples: {missing[:5]}")
    info = model.load_state_dict(state_dict, strict=False, assign=True)
    if info.missing_keys or info.unexpected_keys:
        raise RuntimeError(f"Strict NVFP4/AWQ load failed: {info}")
    logger.info("Attached %d Comfy-quantized H3 layers: %s", len(marker_keys), format_counts)
    return len(marker_keys)
