import torch

from racer import bitmatrix, gf256


def test_element_bitmatrix_matches_gf_multiply():
    for coef in [0, 1, 2, 7, 142, 244]:
        bm = bitmatrix.coeff_to_bitmatrix(coef)
        for value in [0, 1, 3, 17, 255]:
            assert bitmatrix.apply_bitmatrix_to_byte(bm, value) == gf256.gf_mul(coef, value)


def test_all_coefficients_and_bytes_match_gf_multiply():
    for coef in range(256):
        for value in range(256):
            assert bitmatrix.apply_bitmatrix_cpu(coef, value) == gf256.gf_mul(coef, value)


def test_zero_and_one_bitmatrices():
    zero = bitmatrix.coeff_to_bitmatrix(0)
    one = bitmatrix.coeff_to_bitmatrix(1)
    assert bitmatrix.bitmatrix_weight(zero) == 0
    assert one == [[1 if i == j else 0 for j in range(8)] for i in range(8)]


def test_matrix_to_bitmatrix_shape():
    bm = bitmatrix.matrix_to_bitmatrix([[1, 2], [3, 4]])
    assert len(bm) == 16
    assert all(len(row) == 16 for row in bm)
    assert bitmatrix.bitmatrix_weight(bm) > 0


def test_apply_bitmatrix_cpu_tensor():
    src = torch.tensor([0, 1, 2, 255], dtype=torch.uint8)
    out = bitmatrix.apply_bitmatrix_cpu(7, src)
    expected = torch.tensor([gf256.gf_mul(7, int(v)) for v in src], dtype=torch.uint8)
    assert torch.equal(out, expected)
