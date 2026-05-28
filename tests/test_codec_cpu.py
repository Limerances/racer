from itertools import combinations

import numpy as np
import torch

from racer import cauchy, codec_cpu


def test_cpu_encode_decode_single_failure():
    matrix = cauchy.systematic_matrix(3, 1)
    data = [
        torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
        torch.tensor([5, 6, 7, 8], dtype=torch.uint8),
        torch.tensor([9, 10, 11, 12], dtype=torch.uint8),
    ]
    code = codec_cpu.encode_blocks(data, matrix)
    decoded = codec_cpu.decode_blocks([code[1], code[2], code[3]], [1, 2, 3], matrix)
    for actual, expected in zip(decoded, data):
        assert torch.equal(actual, expected)


def test_cpu_encode_decode_two_failures_k2_m2():
    matrix = cauchy.systematic_matrix(2, 2)
    data = [
        torch.arange(0, 16, dtype=torch.uint8),
        torch.arange(16, 32, dtype=torch.uint8),
    ]
    code = codec_cpu.encode_blocks(data, matrix)
    decoded = codec_cpu.decode_blocks([code[2], code[3]], [2, 3], matrix)
    assert torch.equal(decoded[0], data[0])
    assert torch.equal(decoded[1], data[1])


def test_encode_cpu_pads_unequal_torch_buffers():
    matrix = cauchy.generate_systematic_matrix(3, 1)
    data = [
        torch.arange(0, 5, dtype=torch.uint8),
        torch.arange(10, 18, dtype=torch.uint8),
        torch.arange(30, 33, dtype=torch.uint8),
    ]
    code = codec_cpu.encode_cpu(data, matrix)
    assert all(chunk.numel() == 8 for chunk in code)
    decoded = codec_cpu.decode_cpu([code[1], code[2], code[3]], [1, 2, 3], matrix)
    for actual, expected in zip(decoded, data):
        assert torch.equal(actual[: expected.numel()], expected)
        assert torch.equal(actual[expected.numel() :], torch.zeros(8 - expected.numel(), dtype=torch.uint8))


def test_encode_cpu_accepts_numpy_uint8_buffers():
    matrix = cauchy.generate_systematic_matrix(2, 1)
    data = [
        np.arange(0, 7, dtype=np.uint8),
        np.arange(20, 29, dtype=np.uint8),
    ]
    code = codec_cpu.encode_cpu(data, matrix)
    decoded = codec_cpu.decode_cpu([code[0], code[2]], [0, 2], matrix)
    assert isinstance(decoded[0], np.ndarray)
    assert np.array_equal(decoded[0][: data[0].size], data[0])
    assert np.array_equal(decoded[1][: data[1].size], data[1])


def test_k3_m1_enumerates_all_single_chunk_failures():
    k, m = 3, 1
    E = cauchy.generate_systematic_matrix(k, m)
    data = [torch.randint(0, 256, (257,), dtype=torch.uint8) for _ in range(k)]
    code = codec_cpu.encode_cpu(data, E)
    for failed in range(k + m):
        survivors = [row for row in range(k + m) if row != failed]
        decoded = codec_cpu.decode_cpu([code[row] for row in survivors], survivors, E)
        for actual, expected in zip(decoded, data):
            assert torch.equal(actual, expected)


def test_k2_m2_enumerates_all_double_chunk_failures():
    k, m = 2, 2
    E = cauchy.generate_systematic_matrix(k, m)
    data = [torch.randint(0, 256, (513,), dtype=torch.uint8) for _ in range(k)]
    code = codec_cpu.encode_cpu(data, E)
    for failed_rows in combinations(range(k + m), m):
        survivors = [row for row in range(k + m) if row not in failed_rows]
        decoded = codec_cpu.decode_cpu([code[row] for row in survivors], survivors, E)
        for actual, expected in zip(decoded, data):
            assert torch.equal(actual, expected)


def test_k2_m2_decodes_with_fewer_than_m_failures():
    k, m = 2, 2
    E = cauchy.generate_systematic_matrix(k, m)
    data = [torch.randint(0, 256, (129,), dtype=torch.uint8) for _ in range(k)]
    code = codec_cpu.encode_cpu(data, E)
    for num_failed in range(m):
        for failed_rows in combinations(range(k + m), num_failed):
            survivors = [row for row in range(k + m) if row not in failed_rows]
            decoded = codec_cpu.decode_cpu([code[row] for row in survivors], survivors, E)
            for actual, expected in zip(decoded, data):
                assert torch.equal(actual, expected)
