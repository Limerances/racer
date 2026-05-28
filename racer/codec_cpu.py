"""CPU reference codec for byte-exact GF(2^8) encode/decode.

The implementation is intentionally simple and exact: inputs are padded to a
common byte length, each byte is treated as one GF(2^8) symbol, and matrix
application uses XOR of table-based GF products.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from . import gf256
from .cauchy import get_submatrix_rows

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected in normal test/dev envs.
    np = None  # type: ignore[assignment]


def _is_numpy_array(value: Any) -> bool:
    return np is not None and isinstance(value, np.ndarray)


def _as_cpu_uint8_flat(chunk: Any) -> torch.Tensor:
    if isinstance(chunk, torch.Tensor):
        if chunk.dtype != torch.uint8:
            raise TypeError("CPU codec chunks must be torch.uint8")
        if chunk.device.type != "cpu":
            raise ValueError("CPU codec expects CPU tensors")
        return chunk.contiguous().view(-1)
    if _is_numpy_array(chunk):
        if chunk.dtype != np.uint8:  # type: ignore[union-attr]
            raise TypeError("CPU codec chunks must be numpy uint8 arrays")
        arr = np.ascontiguousarray(chunk.reshape(-1))  # type: ignore[union-attr]
        return torch.from_numpy(arr)
    raise TypeError("CPU codec chunks must be torch.Tensor or numpy.ndarray")


def _restore_type(flat: torch.Tensor, like: Any) -> torch.Tensor | Any:
    if isinstance(like, torch.Tensor):
        return flat.clone()
    if _is_numpy_array(like):
        return flat.numpy().copy()
    raise TypeError("unsupported output type")


def _pad_inputs(inputs: Sequence[Any]) -> tuple[list[torch.Tensor], int, Any]:
    if not inputs:
        raise ValueError("inputs must be non-empty")
    flat_inputs = [_as_cpu_uint8_flat(chunk) for chunk in inputs]
    max_len = max(int(chunk.numel()) for chunk in flat_inputs)
    padded = []
    for chunk in flat_inputs:
        if chunk.numel() == max_len:
            padded.append(chunk.clone())
        else:
            out = torch.zeros(max_len, dtype=torch.uint8)
            out[: chunk.numel()].copy_(chunk)
            padded.append(out)
    return padded, max_len, inputs[0]


def _validate_blocks(blocks: Sequence[torch.Tensor], device_type: str = "cpu") -> None:
    if not blocks:
        raise ValueError("blocks must be non-empty")
    size = blocks[0].numel()
    for block in blocks:
        if block.dtype != torch.uint8:
            raise TypeError("codec blocks must be torch.uint8")
        if block.device.type != device_type:
            raise ValueError(f"expected {device_type} tensors")
        if block.numel() != size:
            raise ValueError("all codec blocks must have the same numel")


def apply_matrix_cpu(inputs: Sequence[Any], coeff_matrix: Sequence[Sequence[int]]) -> list[Any]:
    padded, _, first = _pad_inputs(inputs)
    if not coeff_matrix:
        raise ValueError("coeff_matrix must be non-empty")
    width = len(coeff_matrix[0])
    if width != len(padded):
        raise ValueError("matrix width must equal number of input chunks")
    for row in coeff_matrix:
        if len(row) != width:
            raise ValueError("coeff_matrix must be rectangular")

    outputs: list[Any] = []
    for row in coeff_matrix:
        out = torch.zeros_like(padded[0])
        for coef, chunk in zip(row, padded):
            c = int(coef) & 0xFF
            if c:
                out.bitwise_xor_(gf256.mul_tensor(chunk, c))
        outputs.append(_restore_type(out, first))
    return outputs


def encode_cpu(data_chunks: Sequence[Any], E_or_C: Sequence[Sequence[int]]) -> list[Any]:
    return apply_matrix_cpu(data_chunks, E_or_C)


def decode_cpu(
    survivor_chunks: Sequence[Any],
    survivor_rows: Sequence[int],
    E: Sequence[Sequence[int]],
) -> list[Any]:
    if not E:
        raise ValueError("E must be non-empty")
    k = len(E[0])
    if len(survivor_chunks) != len(survivor_rows):
        raise ValueError("survivor_chunks and survivor_rows length mismatch")
    if len(survivor_rows) < k:
        raise ValueError(f"need at least k={k} survivors to decode")
    chosen_rows = list(survivor_rows[:k])
    chosen_chunks = list(survivor_chunks[:k])
    A = get_submatrix_rows(E, chosen_rows)
    A_inv = gf256.gf_matrix_inverse(A)
    return apply_matrix_cpu(chosen_chunks, A_inv)


def encode_blocks(
    data_blocks: Sequence[torch.Tensor],
    matrix: Sequence[Sequence[int]],
) -> list[torch.Tensor]:
    outputs = encode_cpu(data_blocks, matrix)
    return [out for out in outputs if isinstance(out, torch.Tensor)]


def decode_blocks(
    code_blocks: Sequence[torch.Tensor],
    survivor_rows: Sequence[int],
    encode_matrix: Sequence[Sequence[int]],
) -> list[torch.Tensor]:
    outputs = decode_cpu(code_blocks, survivor_rows, encode_matrix)
    return [out for out in outputs if isinstance(out, torch.Tensor)]
