"""CUDA codec wrappers for RACER GF(2^8) primitives.

GF work is intentionally routed through the compiled CUDA extension. Large
checkpoint buffers must not fall back to host or CPU implementations; only small
coefficient metadata and timing scalars cross to host.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch

from . import gf256


@lru_cache(maxsize=1)
def _extension():
    try:
        from . import _C  # type: ignore

        return _C
    except Exception:
        pass

    if os.environ.get("RACER_JIT_COMPILE") != "1":
        return None

    try:
        from torch.utils.cpp_extension import load
    except Exception:
        return None

    root = Path(__file__).resolve().parent
    try:
        return load(
            name="racer_jit_C",
            sources=[str(root / "csrc" / "binding.cpp"), str(root / "csrc" / "racer_cuda.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        return None


def extension_available() -> bool:
    return _extension() is not None


def _extension_function(name: str):
    ext = _extension()
    if ext is None or not hasattr(ext, name):
        raise RuntimeError(
            f"RACER CUDA extension function {name!r} is required; rebuild the package "
            "or set RACER_JIT_COMPILE=1 in a CUDA build environment"
        )
    return getattr(ext, name)


def _sync(device: torch.device) -> None:
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(index)


def _validate_blocks(blocks: Sequence[torch.Tensor]) -> None:
    if not blocks:
        raise ValueError("blocks must be non-empty")
    size = blocks[0].numel()
    device = blocks[0].device
    for block in blocks:
        if block.dtype != torch.uint8:
            raise TypeError("codec blocks must be torch.uint8")
        if block.device.type != "cuda":
            raise ValueError("codec_cuda expects CUDA tensors")
        if block.device != device:
            raise ValueError("all CUDA codec blocks must be on the same device")
        if block.numel() != size:
            raise ValueError("all codec blocks must have the same numel")
        if not block.is_contiguous():
            raise ValueError("all CUDA codec blocks must be contiguous")


def _validate_cuda_buffer(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise TypeError(f"{name} must be torch.uint8")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def gf256_mul(src_uint8_cuda: torch.Tensor, coeff: int, *, synchronize: bool = False) -> torch.Tensor:
    src = _validate_cuda_buffer(src_uint8_cuda, "src_uint8_cuda")
    c = int(coeff) & 0xFF
    out = _extension_function("gf256_mul")(src, c)
    if synchronize:
        _sync(src.device)
    return out


def gf256_mul_xor(
    src_uint8_cuda: torch.Tensor,
    dst_uint8_cuda: torch.Tensor,
    coeff: int,
    *,
    synchronize: bool = False,
) -> torch.Tensor:
    src = _validate_cuda_buffer(src_uint8_cuda, "src_uint8_cuda")
    dst = _validate_cuda_buffer(dst_uint8_cuda, "dst_uint8_cuda")
    if src.device != dst.device:
        raise ValueError("src and dst must be on the same CUDA device")
    if src.numel() != dst.numel():
        raise ValueError("src and dst must have the same numel")
    c = int(coeff) & 0xFF
    out = _extension_function("gf256_mul_xor")(src, dst, c)
    if synchronize:
        _sync(src.device)
    return out


def xor_inplace(
    dst_uint8_cuda: torch.Tensor,
    src_uint8_cuda: torch.Tensor,
    *,
    synchronize: bool = False,
) -> torch.Tensor:
    dst = _validate_cuda_buffer(dst_uint8_cuda, "dst_uint8_cuda")
    src = _validate_cuda_buffer(src_uint8_cuda, "src_uint8_cuda")
    if src.device != dst.device:
        raise ValueError("src and dst must be on the same CUDA device")
    if src.numel() != dst.numel():
        raise ValueError("src and dst must have the same numel")
    out = _extension_function("xor_inplace")(dst, src)
    if synchronize:
        _sync(dst.device)
    return out



def matmul(data: torch.Tensor, coeff: torch.Tensor, *, synchronize: bool = False) -> torch.Tensor:
    if data.dtype != torch.uint8 or coeff.dtype != torch.uint8:
        raise TypeError("CUDA GF matmul expects uint8 tensors")
    if data.device.type != "cuda" or coeff.device.type != "cuda":
        raise ValueError("CUDA GF matmul expects CUDA tensors")
    data = data.contiguous().view(data.shape[0], -1)
    coeff = coeff.contiguous()
    out = _extension_function("gf256_matmul")(data, coeff)
    if synchronize:
        _sync(data.device)
    return out


def apply_matrix_cuda(
    inputs: Sequence[torch.Tensor],
    coeff_matrix: Sequence[Sequence[int]] | torch.Tensor,
    outputs: Sequence[torch.Tensor] | None = None,
    *,
    synchronize: bool = False,
) -> list[torch.Tensor]:
    _validate_blocks(inputs)
    device = inputs[0].device
    flat_inputs = [tensor.view(-1) for tensor in inputs]
    if isinstance(coeff_matrix, torch.Tensor):
        coeff = coeff_matrix.to(device=device, dtype=torch.uint8).contiguous()
    else:
        coeff = gf256.coefficients_to_tensor(coeff_matrix, device)
    if coeff.dim() != 2 or coeff.shape[1] != len(flat_inputs):
        raise ValueError("coeff_matrix must have shape [num_outputs, num_inputs]")

    if outputs is None:
        flat_outputs = [torch.empty_like(flat_inputs[0]) for _ in range(int(coeff.shape[0]))]
    else:
        if len(outputs) != int(coeff.shape[0]):
            raise ValueError("outputs length must equal coeff_matrix row count")
        flat_outputs = [_validate_cuda_buffer(out, "output").view(-1) for out in outputs]
        for out in flat_outputs:
            if out.device != device:
                raise ValueError("all outputs must be on the same CUDA device as inputs")
            if out.numel() != flat_inputs[0].numel():
                raise ValueError("all outputs must have the same numel as inputs")

    use_table_kernel = os.environ.get("RACER_USE_TABLE_KERNEL") == "1"
    if outputs is None and use_table_kernel:
        stacked = torch.stack(flat_inputs, dim=0).contiguous()
        mul_table = gf256.torch_mul_table(device)
        table_result = _extension_function("apply_matrix_cuda_table")(stacked, coeff, mul_table)
        result = [table_result[row].contiguous().view(-1) for row in range(int(table_result.shape[0]))]
    else:
        result = list(_extension_function("apply_matrix_cuda")(flat_inputs, coeff, flat_outputs))
    if synchronize:
        _sync(device)
    return result


def benchmark_apply_matrix_cuda(
    inputs: Sequence[torch.Tensor],
    coeff_matrix: Sequence[Sequence[int]] | torch.Tensor,
    iters: int = 10,
    warmup: int = 2,
) -> dict[str, float]:
    if iters <= 0:
        raise ValueError("iters must be positive")
    _validate_blocks(inputs)
    device = inputs[0].device
    for _ in range(warmup):
        apply_matrix_cuda(inputs, coeff_matrix)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        apply_matrix_cuda(inputs, coeff_matrix)
    end.record()
    torch.cuda.synchronize(device)
    elapsed_ms = start.elapsed_time(end)
    nbytes = sum(t.numel() for t in inputs) * iters
    gib = nbytes / (1024**3)
    seconds = elapsed_ms / 1000.0
    return {"seconds": seconds, "input_gib": gib, "input_gib_per_s": gib / seconds}


def encode_blocks(
    data_blocks: Sequence[torch.Tensor],
    matrix: Sequence[Sequence[int]],
) -> list[torch.Tensor]:
    return apply_matrix_cuda(data_blocks, matrix)


def decode_blocks(
    code_blocks: Sequence[torch.Tensor],
    survivor_rows: Sequence[int],
    encode_matrix: Sequence[Sequence[int]],
) -> list[torch.Tensor]:
    _validate_blocks(code_blocks)
    if len(code_blocks) != len(survivor_rows):
        raise ValueError("code_blocks and survivor_rows length mismatch")
    k = len(encode_matrix[0])
    if len(survivor_rows) < k:
        raise ValueError(f"need at least k={k} survivors to decode")
    chosen_rows = list(survivor_rows[:k])
    chosen_blocks = list(code_blocks[:k])
    selected = gf256.select_rows(encode_matrix, chosen_rows)
    inverse = gf256.invert_matrix(selected)
    return encode_blocks(chosen_blocks, inverse)
