import pytest
import torch

from racer import bitmatrix, codec_cuda, gf256


def test_element_bitmatrix_matches_gf_multiply():
    for coef in [0, 1, 2, 7, 142, 244]:
        bm = bitmatrix.coeff_to_bitmatrix(coef)
        for value in [0, 1, 3, 17, 255]:
            assert bitmatrix.apply_bitmatrix_to_byte(bm, value) == gf256.gf_mul(coef, value)


def test_all_coefficients_and_bytes_match_gf_multiply():
    for coef in range(256):
        bm = bitmatrix.coeff_to_bitmatrix(coef)
        for value in range(256):
            assert bitmatrix.apply_bitmatrix_to_byte(bm, value) == gf256.gf_mul(coef, value)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bitmatrix_kernel_matches_gf_matrix_kernel():
    if codec_cuda._optional_extension_function("apply_bitmatrix_cuda") is None:
        pytest.skip("RACER CUDA bitmatrix extension is unavailable")
    inputs = [
        (torch.arange(257, dtype=torch.int16, device="cuda") % 256).to(torch.uint8),
        ((torch.arange(257, dtype=torch.int16, device="cuda") + 17) % 256).to(torch.uint8),
        ((torch.arange(257, dtype=torch.int16, device="cuda") + 31) % 256).to(torch.uint8),
    ]
    matrix = [[1, 2, 3], [5, 0, 7]]

    expected = codec_cuda.apply_matrix_cuda(inputs, matrix, synchronize=True)
    actual = bitmatrix.apply_matrix_cuda(inputs, matrix, synchronize=True)

    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)
