import pytest
import torch

from racer import cauchy, codec_cpu, codec_cuda, gf256
from racer.layout import ElasticLayout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_encode_decode_matches_cpu_path():
    device = torch.device("cuda:0")
    matrix = cauchy.systematic_matrix(3, 1)
    data = [
        torch.arange(0, 1024, dtype=torch.uint8, device=device),
        torch.arange(31, 1055, dtype=torch.uint8, device=device),
        torch.arange(99, 1123, dtype=torch.uint8, device=device),
    ]
    code = codec_cuda.encode_blocks(data, matrix)
    decoded = codec_cuda.decode_blocks([code[0], code[2], code[3]], [0, 2, 3], matrix)
    for actual, expected in zip(decoded, data):
        assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("nbytes", [1024, 1024 * 1024])
def test_cuda_gf256_mul_matches_cpu_table(nbytes):
    device = torch.device("cuda:0")
    src = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=device)
    coeff = 173
    out = codec_cuda.gf256_mul(src, coeff)
    table = torch.tensor(gf256.MUL_TABLE[coeff], dtype=torch.uint8)
    expected = table[src.cpu().long()]
    assert torch.equal(out.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_mul_xor_and_xor_inplace():
    device = torch.device("cuda:0")
    src = torch.randint(0, 256, (4096,), dtype=torch.uint8, device=device)
    dst = torch.randint(0, 256, (4096,), dtype=torch.uint8, device=device)
    original = dst.clone()
    coeff = 29
    codec_cuda.gf256_mul_xor(src, dst, coeff)
    table = torch.tensor(gf256.MUL_TABLE[coeff], dtype=torch.uint8)
    expected = original.cpu().bitwise_xor(table[src.cpu().long()])
    assert torch.equal(dst.cpu(), expected)

    other = torch.randint(0, 256, (4096,), dtype=torch.uint8, device=device)
    before = dst.clone()
    codec_cuda.xor_inplace(dst, other)
    assert torch.equal(dst.cpu(), before.cpu().bitwise_xor(other.cpu()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("k,m,failed_rows", [(3, 1, [1]), (2, 2, [0, 3])])
def test_cuda_apply_matrix_encode_decode_matches_cpu(k, m, failed_rows):
    device = torch.device("cuda:0")
    E = cauchy.generate_systematic_matrix(k, m)
    data_cpu = [torch.randint(0, 256, (4097,), dtype=torch.uint8) for _ in range(k)]
    data_cuda = [chunk.to(device) for chunk in data_cpu]

    code_cuda = codec_cuda.apply_matrix_cuda(data_cuda, E)
    code_cpu = codec_cpu.encode_cpu(data_cpu, E)
    for actual, expected in zip(code_cuda, code_cpu):
        assert torch.equal(actual.cpu(), expected)

    survivors = [row for row in range(k + m) if row not in failed_rows]
    decoded_cuda = codec_cuda.decode_blocks([code_cuda[row] for row in survivors], survivors, E)
    decoded_cpu = codec_cpu.decode_cpu([code_cpu[row] for row in survivors], survivors, E)
    for actual, expected in zip(decoded_cuda, decoded_cpu):
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_decode_accepts_more_than_k_survivors():
    device = torch.device("cuda:0")
    k, m = 2, 2
    E = cauchy.generate_systematic_matrix(k, m)
    data_cpu = [torch.randint(0, 256, (1025,), dtype=torch.uint8) for _ in range(k)]
    data_cuda = [chunk.to(device) for chunk in data_cpu]
    code_cuda = codec_cuda.apply_matrix_cuda(data_cuda, E)
    code_cpu = codec_cpu.encode_cpu(data_cpu, E)
    survivors = [0, 2, 3]
    decoded_cuda = codec_cuda.decode_blocks([code_cuda[row] for row in survivors], survivors, E)
    decoded_cpu = codec_cpu.decode_cpu([code_cpu[row] for row in survivors], survivors, E)
    for actual, expected in zip(decoded_cuda, decoded_cpu):
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_virtual_zero_layout_recovery():
    device = torch.device("cuda:0")
    layout = ElasticLayout.build([0, 1, 2, 3], [4], k=3, m=1)
    E = cauchy.generate_systematic_matrix(3, 1)
    packets = {
        rank: torch.randint(0, 256, (2048,), dtype=torch.uint8, device=device)
        for rank in layout.train_ranks
    }

    for group in layout.reduction_groups:
        chunks = [
            torch.zeros(2048, dtype=torch.uint8, device=device) if slot.is_virtual_zero else packets[slot.train_rank]
            for slot in group
        ]
        code = codec_cuda.apply_matrix_cuda(chunks, E)
        survivors = [1, 2, 3]
        decoded = codec_cuda.decode_blocks([code[row] for row in survivors], survivors, E)
        for col, slot in enumerate(group):
            if not slot.is_virtual_zero:
                assert torch.equal(decoded[col].cpu(), packets[slot.train_rank].cpu())
