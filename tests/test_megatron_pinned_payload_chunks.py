from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


MEGATRON_ROOT = Path("/workspace/Megatron-LM-FT")
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pinned_payload_chunks_match_cuda_pack() -> None:
    from megatron.training.racer.tensor_tree import (
        PinnedPayloadBufferPool,
        pack_tensor_chunks,
        pack_tensor_chunks_to_pinned,
    )

    device = torch.device("cuda:0")
    tensors = [
        torch.arange(0, 3000, dtype=torch.int32, device=device),
        torch.arange(0, 2500, dtype=torch.float16, device=device),
    ]
    chunk_size = 4096

    chunks, cuda_payloads = pack_tensor_chunks(tensors, device, chunk_size)
    pool = PinnedPayloadBufferPool()
    pinned_payloads = pack_tensor_chunks_to_pinned(
        tensors,
        device,
        chunk_size,
        host_buffer_pool=pool,
        expected_chunk_count=len(cuda_payloads),
    )

    assert pinned_payloads.chunks == chunks
    assert len(pinned_payloads) == len(cuda_payloads)
    assert pinned_payloads.total_reserved_nbytes == chunk_size * len(cuda_payloads)
    assert pinned_payloads.total_valid_nbytes == sum(int(payload.numel()) for payload in cuda_payloads)
    assert all(buffer.is_pinned() for buffer in pinned_payloads.buffers)

    for index, expected in enumerate(cuda_payloads):
        actual = pinned_payloads.cuda_payload(index, device)
        torch.cuda.synchronize(device)
        assert int(actual.numel()) == int(expected.numel())
        assert torch.equal(actual, expected)

    first_ptrs = [int(buffer.data_ptr()) for buffer in pinned_payloads.buffers]
    pinned_payloads_second = pack_tensor_chunks_to_pinned(
        tensors,
        device,
        chunk_size,
        host_buffer_pool=pool,
        expected_chunk_count=len(cuda_payloads),
    )
    second_ptrs = [int(buffer.data_ptr()) for buffer in pinned_payloads_second.buffers]
    assert second_ptrs == first_ptrs
    assert pinned_payloads_second.profile["payload_pool_allocated_buffers"] == 0.0
    assert pinned_payloads_second.profile["payload_pool_reused_buffers"] == float(len(cuda_payloads))
