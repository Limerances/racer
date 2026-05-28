import pytest
import random

from racer import gf256


def _random_invertible_matrix(n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    for _ in range(64):
        op = rng.randrange(3)
        a = rng.randrange(n)
        b = rng.randrange(n)
        if op == 0 and a != b:
            matrix[a], matrix[b] = matrix[b], matrix[a]
        elif op == 1:
            factor = rng.randrange(1, 256)
            matrix[a] = [gf256.gf_mul(factor, value) for value in matrix[a]]
        elif op == 2 and a != b:
            factor = rng.randrange(1, 256)
            matrix[a] = [
                value ^ gf256.gf_mul(factor, other)
                for value, other in zip(matrix[a], matrix[b])
            ]
    return matrix


def test_gf_add_and_sub_are_xor():
    for a in range(256):
        for b in range(256):
            assert gf256.gf_add(a, b) == (a ^ b)
            assert gf256.gf_sub(a, b) == (a ^ b)


def test_mul_identity_and_zero():
    for x in range(256):
        assert gf256.gf_mul(x, 0) == 0
        assert gf256.gf_mul(x, 1) == x
        assert gf256.gf_mul(1, x) == x


def test_inverse_and_division():
    for x in range(1, 256):
        inv = gf256.gf_inv(x)
        assert gf256.gf_mul(x, inv) == 1
        assert gf256.gf_div(x, x) == 1
        assert gf256.gf_div(gf256.gf_mul(x, 7), 7) == x
    with pytest.raises(ZeroDivisionError):
        gf256.gf_inv(0)


def test_matrix_inverse():
    matrix = [[1, 0, 0], [1, 122, 244], [122, 71, 173]]
    inv = gf256.gf_matrix_inverse(matrix)
    ident = gf256.gf_matmul(matrix, inv)
    assert ident == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_random_invertible_matrix_inverse_is_exact():
    for n in [2, 3, 4, 6]:
        matrix = _random_invertible_matrix(n, seed=n)
        inv = gf256.gf_matrix_inverse(matrix)
        ident = gf256.gf_matmul(matrix, inv)
        assert ident == [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def test_lookup_tables_have_byte_shape():
    assert len(gf256.MUL_TABLE) == 256
    assert all(len(row) == 256 for row in gf256.MUL_TABLE)
    assert len(gf256.INV_TABLE) == 256
