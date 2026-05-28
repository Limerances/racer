"""Exact GF(2^8) arithmetic used by RACER codecs.

The field uses primitive polynomial 0x11d, matching the Cauchy Reed-Solomon
setup used by Jerasure-style checkpoint coding. All operations are integer
operations over bytes; there is no floating-point arithmetic in the coding
path.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import torch

PRIMITIVE_POLY = 0x11D
FIELD_SIZE = 256
FIELD_ORDER = 255


def _mul_no_table(a: int, b: int) -> int:
    a &= 0xFF
    b &= 0xFF
    p = 0
    for _ in range(8):
        if b & 1:
            p ^= a
        carry = a & 0x80
        a = (a << 1) & 0xFF
        if carry:
            a ^= PRIMITIVE_POLY & 0xFF
        b >>= 1
    return p


def _build_tables() -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    exp = [0] * (FIELD_ORDER * 2)
    log = [0] * FIELD_SIZE
    x = 1
    for i in range(FIELD_ORDER):
        exp[i] = x
        log[x] = i
        x = _mul_no_table(x, 2)
    for i in range(FIELD_ORDER, FIELD_ORDER * 2):
        exp[i] = exp[i - FIELD_ORDER]

    mul_table: list[list[int]] = [[0] * FIELD_SIZE for _ in range(FIELD_SIZE)]
    inv_table = [0] * FIELD_SIZE
    for a in range(FIELD_SIZE):
        for b in range(FIELD_SIZE):
            if a == 0 or b == 0:
                v = 0
            else:
                v = exp[log[a] + log[b]]
            mul_table[a][b] = v
    for a in range(1, FIELD_SIZE):
        inv_table[a] = exp[FIELD_ORDER - log[a]]
    flat_mul_table = [mul_table[a][b] for a in range(FIELD_SIZE) for b in range(FIELD_SIZE)]
    return tuple(exp), tuple(log), tuple(tuple(row) for row in mul_table), tuple(inv_table), tuple(flat_mul_table)


EXP_TABLE, LOG_TABLE, MUL_TABLE, INV_TABLE, FLAT_MUL_TABLE = _build_tables()


def add(a: int, b: int) -> int:
    return (a ^ b) & 0xFF


def sub(a: int, b: int) -> int:
    return add(a, b)


def mul(a: int, b: int) -> int:
    a &= 0xFF
    b &= 0xFF
    return MUL_TABLE[a][b]


def inverse(a: int) -> int:
    a &= 0xFF
    if a == 0:
        raise ZeroDivisionError("0 has no multiplicative inverse in GF(2^8)")
    return INV_TABLE[a]


def div(a: int, b: int) -> int:
    a &= 0xFF
    b &= 0xFF
    if b == 0:
        raise ZeroDivisionError("division by zero in GF(2^8)")
    if a == 0:
        return 0
    return EXP_TABLE[(LOG_TABLE[a] - LOG_TABLE[b]) % FIELD_ORDER]


def pow(a: int, n: int) -> int:
    a &= 0xFF
    if n < 0:
        return pow(inverse(a), -n)
    out = 1
    while n:
        if n & 1:
            out = mul(out, a)
        a = mul(a, a)
        n >>= 1
    return out


def gf_add(a: int, b: int) -> int:
    return add(a, b)


def gf_sub(a: int, b: int) -> int:
    return sub(a, b)


def gf_mul(a: int, b: int) -> int:
    return mul(a, b)


def gf_pow(a: int, p: int) -> int:
    return pow(a, p)


def gf_inv(a: int) -> int:
    return inverse(a)


def gf_div(a: int, b: int) -> int:
    return div(a, b)


def _check_matrix(matrix: Sequence[Sequence[int]]) -> None:
    if not matrix:
        raise ValueError("matrix must be non-empty")
    width = len(matrix[0])
    if width == 0:
        raise ValueError("matrix rows must be non-empty")
    for row in matrix:
        if len(row) != width:
            raise ValueError("matrix must be rectangular")
        for v in row:
            if not 0 <= int(v) < FIELD_SIZE:
                raise ValueError(f"matrix coefficient out of GF(2^8): {v}")


def matmul(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> list[list[int]]:
    _check_matrix(a)
    _check_matrix(b)
    if len(a[0]) != len(b):
        raise ValueError("matrix dimension mismatch")
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            acc = 0
            for k in range(inner):
                acc ^= mul(int(a[i][k]), int(b[k][j]))
            out[i][j] = acc
    return out


def gf_matmul(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> list[list[int]]:
    return matmul(a, b)


def invert_matrix(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    _check_matrix(matrix)
    n = len(matrix)
    if len(matrix[0]) != n:
        raise ValueError("only square matrices can be inverted")

    aug = [
        [int(matrix[i][j]) & 0xFF for j in range(n)]
        + [1 if i == j else 0 for j in range(n)]
        for i in range(n)
    ]

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            raise ValueError("matrix is singular over GF(2^8)")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]

        inv_pivot = inverse(aug[col][col])
        if inv_pivot != 1:
            for j in range(2 * n):
                aug[col][j] = mul(aug[col][j], inv_pivot)

        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0:
                continue
            for j in range(2 * n):
                aug[row][j] ^= mul(factor, aug[col][j])

    return [row[n:] for row in aug]


def gf_matrix_inverse(matrix: Sequence[Sequence[int]]) -> list[list[int]]:
    return invert_matrix(matrix)


def select_rows(matrix: Sequence[Sequence[int]], rows: Iterable[int]) -> list[list[int]]:
    return [[int(v) & 0xFF for v in matrix[row]] for row in rows]


@lru_cache(maxsize=32)
def _torch_mul_table(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    return torch.tensor(FLAT_MUL_TABLE, dtype=torch.uint8, device=device).view(256, 256)


def torch_mul_table(device: torch.device | str) -> torch.Tensor:
    dev = torch.device(device)
    return _torch_mul_table(dev.type, dev.index)


def mul_tensor(tensor: torch.Tensor, coefficient: int) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise TypeError("GF tensor multiplication expects torch.uint8 tensors")
    coef = int(coefficient) & 0xFF
    if coef == 0:
        return torch.zeros_like(tensor)
    if coef == 1:
        return tensor.clone()
    table = torch_mul_table(tensor.device)
    return table[coef][tensor.to(torch.long)]


def xor_sum(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    if not parts:
        raise ValueError("xor_sum needs at least one tensor")
    out = torch.zeros_like(parts[0])
    for part in parts:
        out.bitwise_xor_(part)
    return out


def coefficients_to_tensor(matrix: Sequence[Sequence[int]], device: torch.device | str) -> torch.Tensor:
    return torch.tensor(matrix, dtype=torch.uint8, device=device).contiguous()
