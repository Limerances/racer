"""Cauchy Reed-Solomon matrix generation for RACER.

`generate_systematic_matrix` returns the encoding matrix E=[I;C]. Its rows map
only to `train_ranks`; spare GPUs are compute resources and are never included
in this matrix.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from . import gf256


def _validate(k: int, m: int, w: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if m <= 0:
        raise ValueError("m must be positive")
    if w != 8:
        raise NotImplementedError("RACER currently implements GF(2^8), so w must be 8")
    if k + m > (1 << w):
        raise ValueError("Cauchy matrix requires k + m <= 2^w")


def cauchy_original_coding_matrix(k: int, m: int, w: int = 8) -> list[list[int]]:
    """Return Jerasure-compatible original Cauchy coding rows, shape m x k."""
    _validate(k, m, w)
    matrix: list[list[int]] = []
    for i in range(m):
        row = []
        for j in range(k):
            denominator = i ^ (m + j)
            row.append(gf256.div(1, denominator))
        matrix.append(row)
    return matrix


def _cauchy_from_xy(x_values: Sequence[int], y_values: Sequence[int], w: int) -> list[list[int]]:
    if w != 8:
        raise NotImplementedError("only w=8 is implemented")
    if len(set(x_values)) != len(x_values):
        raise ValueError("Cauchy x values must be unique")
    if len(set(y_values)) != len(y_values):
        raise ValueError("Cauchy y values must be unique")
    if set(x_values) & set(y_values):
        raise ValueError("Cauchy x and y values must be disjoint")

    matrix: list[list[int]] = []
    for y in y_values:
        row = []
        for x in x_values:
            row.append(gf256.gf_inv(int(x) ^ int(y)))
        matrix.append(row)
    return matrix


def _default_xy(k: int, m: int) -> tuple[list[int], list[int]]:
    # This matches Jerasure's original Cauchy construction:
    # C[i][j] = inv(i ^ (m + j)).
    return list(range(m, m + k)), list(range(m))


def _bitmatrix_cost(matrix: list[list[int]], w: int) -> int:
    from .bitmatrix import bitmatrix_weight, matrix_to_bitmatrix

    return bitmatrix_weight(matrix_to_bitmatrix(matrix, w))


def _random_cauchy_candidate(k: int, m: int, w: int, rng: random.Random) -> list[list[int]]:
    values = list(range(1 << w))
    selected = rng.sample(values, k + m)
    x_values = selected[:k]
    y_values = selected[k:]
    return _cauchy_from_xy(x_values, y_values, w)


def generate_cauchy_matrix(
    k: int,
    m: int,
    w: int = 8,
    optimize: bool = False,
    seed: int = 0,
) -> list[list[int]]:
    _validate(k, m, w)
    x_values, y_values = _default_xy(k, m)
    best = _cauchy_from_xy(x_values, y_values, w)
    if not optimize:
        return best

    rng = random.Random(seed)
    best_cost = _bitmatrix_cost(best, w)
    # A small deterministic search is enough for phase 1. It keeps matrix
    # generation cheap while making optimize=True observable and reproducible.
    for _ in range(128):
        candidate = _random_cauchy_candidate(k, m, w, rng)
        cost = _bitmatrix_cost(candidate, w)
        if cost < best_cost:
            best = candidate
            best_cost = cost
    return best


def _n_ones(value: int, w: int = 8) -> int:
    highbit = 1 << (w - 1)
    pp = gf256.mul(highbit, 2)
    one_masks = [1 << i for i in range(w) if pp & (1 << i)]

    n = value & ((1 << w) - 1)
    cno = n.bit_count()
    total = cno
    for _ in range(1, w):
        if n & highbit:
            n ^= highbit
            n = (n << 1) & ((1 << w) - 1)
            n ^= pp
            cno -= 1
            for mask in one_masks:
                cno += 1 if n & mask else -1
        else:
            n = (n << 1) & ((1 << w) - 1)
        total += cno
    return total


def improve_coding_matrix(matrix: list[list[int]], w: int = 8) -> list[list[int]]:
    """Port of Jerasure's cauchy_improve_coding_matrix for small GF(2^8) matrices."""
    if w != 8:
        raise NotImplementedError("only w=8 is implemented")
    if not matrix:
        return []

    out = [row[:] for row in matrix]
    m = len(out)
    k = len(out[0])

    for col in range(k):
        if out[0][col] != 1:
            factor = gf256.div(1, out[0][col])
            for row in range(m):
                out[row][col] = gf256.mul(out[row][col], factor)

    for row in range(1, m):
        best_ones = sum(_n_ones(v, w) for v in out[row])
        best_col = -1
        for col in range(k):
            if out[row][col] == 1:
                continue
            factor = gf256.div(1, out[row][col])
            candidate = sum(_n_ones(gf256.mul(v, factor), w) for v in out[row])
            if candidate < best_ones:
                best_ones = candidate
                best_col = col
        if best_col != -1:
            factor = gf256.div(1, out[row][best_col])
            for col in range(k):
                out[row][col] = gf256.mul(out[row][col], factor)

    return out


def cauchy_coding_matrix(
    k: int,
    m: int,
    w: int = 8,
    optimize: bool = False,
) -> list[list[int]]:
    return generate_cauchy_matrix(k, m, w, optimize=optimize)


def generate_systematic_matrix(
    k: int,
    m: int,
    w: int = 8,
    optimize: bool = False,
    seed: int = 0,
) -> list[list[int]]:
    _validate(k, m, w)
    identity = [[1 if i == j else 0 for j in range(k)] for i in range(k)]
    return identity + generate_cauchy_matrix(k, m, w, optimize=optimize, seed=seed)


def systematic_matrix(
    k: int,
    m: int,
    w: int = 8,
    optimize: bool = False,
) -> list[list[int]]:
    return generate_systematic_matrix(k, m, w, optimize=optimize)


def get_submatrix_rows(E: Sequence[Sequence[int]], rows: Sequence[int]) -> list[list[int]]:
    if not rows:
        raise ValueError("rows must be non-empty")
    return gf256.select_rows(E, rows)


def make_decoding_matrix(E: Sequence[Sequence[int]], survivor_rows: Sequence[int]) -> list[list[int]]:
    if not E:
        raise ValueError("E must be non-empty")
    k = len(E[0])
    if len(survivor_rows) < k:
        raise ValueError(f"need at least k={k} survivor rows")
    submatrix = get_submatrix_rows(E, survivor_rows[:k])
    return gf256.gf_matrix_inverse(submatrix)
