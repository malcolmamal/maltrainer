"""NVFP4 and INT8 layers that keep their quantized weights and dequantize on use."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

NVFP4_BLOCK_SIZE = 16
_DEQUANT_CHUNK_ELEMENTS = 16 * 1024 * 1024


def unswizzle_nvfp4_scales(scales: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """Convert the cuBLAS 128x4 blocked scale layout to row-major order."""
    row_blocks = (rows + 127) // 128
    column_blocks = (columns + 3) // 4
    padded_rows = row_blocks * 128
    padded_columns = column_blocks * 4
    expected_elements = padded_rows * padded_columns
    if scales.numel() != expected_elements:
        raise ValueError(
            f"Bad NVFP4 scale tensor: expected {expected_elements} elements for {rows}x{columns}, got {scales.numel()}"
        )
    values = scales.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1, 2)
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

    qweight: torch.Tensor
    scales_u8: torch.Tensor

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
            "scales_u8", scales.detach().float().reshape(-1).contiguous().view(torch.uint8), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        lookup_ids = input_ids.reshape(-1).to(self.qweight.device)
        rows = self.qweight.index_select(0, lookup_ids).float()
        scales = self.scales_u8.view(torch.float32).index_select(0, lookup_ids)
        output = (rows * scales.unsqueeze(1)).to(self.output_dtype)
        return output.to(input_ids.device).reshape(*input_ids.shape, self.embedding_dim)


class ComfyNvfp4Linear(nn.Module):
    """NVFP4/AWQ Linear that keeps packed weights and dequantizes each forward."""

    packed_weight: torch.Tensor
    scales_u8: torch.Tensor
    per_tensor_scale_u8: torch.Tensor
    bias: torch.Tensor | None

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
        if packed_weight.dtype is not torch.uint8 or packed_weight.shape != (self.out_features, self.in_features // 2):
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
            blocked_scales.view(torch.float8_e4m3fn), self.out_features, self.in_features // NVFP4_BLOCK_SIZE
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
        value: torch.Tensor | None = getattr(self, "pre_quant_scale_u8", None)
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
