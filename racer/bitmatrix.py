"""Binary expansion of GF(2^8) coefficients.

For a coefficient e, B(e) is an 8x8 bitmatrix such that applying B(e) to the
bits of byte s produces the bits of gf_mul(e, s). V1 uses this for correctness
tests and Cauchy optimization cost estimates.

TODO: compile B(E) into a CUDA XOR schedule kernel so parity can be generated
without table-based GF multiply in future RACER versions.
"""

from __future__ import annotations

from collections.abc import Sequence

from . import gf256


def coeff_to_bitmatrix(e: int, w: int = 8) -> list[list[int]]:
    if w != 8:
        raise NotImplementedError("only w=8 is implemented")
    elem = int(e) & 0xFF
    matrix = [[0 for _ in range(w)] for _ in range(w)]
    for col in range(w):
        product = gf256.gf_mul(elem, 1 << col)
        for row in range(w):
            matrix[row][col] = 1 if product & (1 << row) else 0
    return matrix


def element_to_bitmatrix(element: int, w: int = 8) -> list[list[int]]:
    return coeff_to_bitmatrix(element, w)


def apply_bitmatrix_to_byte(matrix: list[list[int]], value: int, w: int = 8) -> int:
    if w != 8:
        raise NotImplementedError("only w=8 is implemented")
    if len(matrix) != w or any(len(row) != w for row in matrix):
        raise ValueError("bitmatrix must be w x w")
    out = 0
    for row in range(w):
        bit = 0
        for col in range(w):
            if matrix[row][col] and (value & (1 << col)):
                bit ^= 1
        if bit:
            out |= 1 << row
    return out


def matrix_to_bitmatrix(matrix: list[list[int]], w: int = 8) -> list[list[int]]:
    if not matrix:
        raise ValueError("matrix must be non-empty")
    rows = len(matrix)
    cols = len(matrix[0])
    out = [[0 for _ in range(cols * w)] for _ in range(rows * w)]
    for r, row in enumerate(matrix):
        if len(row) != cols:
            raise ValueError("matrix must be rectangular")
        for c, elem in enumerate(row):
            block = coeff_to_bitmatrix(elem, w)
            for br in range(w):
                for bc in range(w):
                    out[r * w + br][c * w + bc] = block[br][bc]
    return out


def bitmatrix_weight(B: Sequence[Sequence[int]]) -> int:
    total = 0
    for row in B:
        total += sum(1 for value in row if int(value) != 0)
    return total
