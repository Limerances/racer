from itertools import combinations

from racer import cauchy
from racer import gf256
from racer.bitmatrix import bitmatrix_weight, matrix_to_bitmatrix


def test_original_cauchy_known_k3_m1():
    assert cauchy.cauchy_original_coding_matrix(3, 1) == [[1, 142, 244]]
    assert cauchy.generate_cauchy_matrix(3, 1) == [[1, 142, 244]]


def test_systematic_shape_and_identity():
    matrix = cauchy.generate_systematic_matrix(2, 2)
    assert len(matrix) == 4
    assert all(len(row) == 2 for row in matrix)
    assert matrix[:2] == [[1, 0], [0, 1]]


def test_all_k_row_submatrices_are_invertible():
    for k, m in [(3, 1), (2, 2), (2, 1), (3, 2), (4, 2)]:
        E = cauchy.generate_systematic_matrix(k, m)
        assert len(E) == k + m
        assert all(len(row) == k for row in E)
        for rows in combinations(range(k + m), k):
            decoding = cauchy.make_decoding_matrix(E, list(rows))
            ident = gf256.gf_matmul(cauchy.get_submatrix_rows(E, list(rows)), decoding)
            assert ident == [[1 if i == j else 0 for j in range(k)] for i in range(k)]


def test_optimized_matrix_is_not_more_expensive_by_bitmatrix_weight():
    k, m = 4, 2
    base = cauchy.generate_cauchy_matrix(k, m, optimize=False)
    optimized = cauchy.generate_cauchy_matrix(k, m, optimize=True, seed=123)
    assert bitmatrix_weight(matrix_to_bitmatrix(optimized)) <= bitmatrix_weight(matrix_to_bitmatrix(base))
