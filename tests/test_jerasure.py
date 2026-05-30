import pytest
import torch

from racer import cauchy, codec_cuda, gf256, jerasure


pytestmark = pytest.mark.skipif(not jerasure.available(), reason="Jerasure library is unavailable")


def _jerasure_apply_matrix(inputs: list[torch.Tensor], matrix: list[list[int]]) -> list[torch.Tensor]:
    outputs = []
    for row in matrix:
        out = torch.zeros_like(inputs[0])
        for coeff, src in zip(row, inputs):
            c = int(coeff) & 0xFF
            if c:
                out.bitwise_xor_(jerasure.region_multiply(src, c))
        outputs.append(out)
    return outputs


def test_jerasure_gf_matches_racer_control_tables():
    for a in range(256):
        for b in [0, 1, 2, 7, 29, 131, 255]:
            assert jerasure.gf_mul(a, b) == gf256.gf_mul(a, b)
    for a in range(1, 256):
        assert jerasure.gf_inv(a) == gf256.gf_inv(a)


def test_jerasure_cauchy_original_matrix_matches_racer_control_matrix():
    for k, m in [(3, 1), (2, 2), (4, 2)]:
        assert jerasure.cauchy_original_coding_matrix(k, m) == cauchy.generate_cauchy_matrix(k, m)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_gf_kernel_matches_jerasure_region_multiply():
    assert codec_cuda.extension_available()
    src_cpu = torch.randint(0, 256, (4096,), dtype=torch.uint8)
    src_cuda = src_cpu.to("cuda:0")
    for coeff in [0, 1, 2, 7, 142, 244]:
        expected = jerasure.region_multiply(src_cpu, coeff)
        actual = codec_cuda.gf256_mul(src_cuda, coeff)
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("k,m", [(3, 1), (2, 2)])
def test_cuda_matrix_kernel_parity_matches_jerasure_matrix_encode(k, m):
    assert codec_cuda.extension_available()
    C = cauchy.generate_cauchy_matrix(k, m)
    E = cauchy.generate_systematic_matrix(k, m)
    data_cpu = [torch.randint(0, 256, (8192,), dtype=torch.uint8) for _ in range(k)]
    data_cuda = [chunk.to("cuda:0") for chunk in data_cpu]

    jerasure_parity = jerasure.matrix_encode(data_cpu, C)
    cuda_code = codec_cuda.apply_matrix_cuda(data_cuda, E)

    for actual, expected in zip(cuda_code[:k], data_cpu):
        assert torch.equal(actual.cpu(), expected)
    for actual, expected in zip(cuda_code[k:], jerasure_parity):
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_decode_kernel_matches_jerasure_decode_matrix_application():
    assert codec_cuda.extension_available()
    k, m = 3, 1
    E = cauchy.generate_systematic_matrix(k, m)
    data_cpu = [torch.randint(0, 256, (4096,), dtype=torch.uint8) for _ in range(k)]
    data_cuda = [chunk.to("cuda:0") for chunk in data_cpu]
    cuda_code = codec_cuda.apply_matrix_cuda(data_cuda, E)
    code_cpu = [row.cpu() for row in cuda_code]

    survivor_rows = [1, 2, 3]
    selected = gf256.select_rows(E, survivor_rows)
    inverse = gf256.invert_matrix(selected)
    expected = _jerasure_apply_matrix([code_cpu[row] for row in survivor_rows], inverse)
    actual = codec_cuda.decode_blocks([cuda_code[row] for row in survivor_rows], survivor_rows, E)

    for actual_chunk, expected_chunk in zip(actual, expected):
        assert torch.equal(actual_chunk.cpu(), expected_chunk)
