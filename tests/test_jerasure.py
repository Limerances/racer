import pytest
import torch

from racer import cauchy, codec_cpu, gf256, jerasure


pytestmark = pytest.mark.skipif(not jerasure.available(), reason="Jerasure library is unavailable")


def test_jerasure_gf_matches_racer_tables():
    for a in range(256):
        for b in [0, 1, 2, 7, 29, 131, 255]:
            assert jerasure.gf_mul(a, b) == gf256.gf_mul(a, b)
    for a in range(1, 256):
        assert jerasure.gf_inv(a) == gf256.gf_inv(a)


def test_jerasure_cauchy_original_matrix_matches_racer():
    for k, m in [(3, 1), (2, 2), (4, 2)]:
        assert jerasure.cauchy_original_coding_matrix(k, m) == cauchy.generate_cauchy_matrix(k, m)


def test_jerasure_region_multiply_matches_racer_cpu():
    src = torch.randint(0, 256, (4096,), dtype=torch.uint8)
    for coeff in [0, 1, 2, 7, 142, 244]:
        assert torch.equal(jerasure.region_multiply(src, coeff), gf256.mul_tensor(src, coeff))


def test_jerasure_matrix_encode_matches_racer_cpu():
    k, m = 3, 1
    C = cauchy.generate_cauchy_matrix(k, m)
    E = cauchy.generate_systematic_matrix(k, m)
    data = [torch.randint(0, 256, (8192,), dtype=torch.uint8) for _ in range(k)]
    jerasure_parity = jerasure.matrix_encode(data, C)
    racer_code = codec_cpu.encode_cpu(data, E)
    assert torch.equal(jerasure_parity[0], racer_code[k])
