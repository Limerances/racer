import pytest
import torch
from dataclasses import replace

from racer import cauchy, distributed, routing
from racer.config import RacerConfig
from racer.layout import ElasticLayout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pad_exact_size_reuses_cuda_payload_storage():
    payload = torch.arange(128, dtype=torch.uint8, device="cuda:0")
    padded = distributed._pad(payload, 128, device=payload.device)

    assert padded.data_ptr() == payload.data_ptr()
    assert torch.equal(padded, payload)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pad_virtual_zero_reuses_cached_cuda_buffer():
    cache = {}
    first = distributed._pad(None, 128, device=torch.device("cuda:0"), zero_cache=cache)
    second = distributed._pad(None, 128, device=torch.device("cuda:0"), zero_cache=cache)

    assert first.data_ptr() == second.data_ptr()
    assert torch.count_nonzero(first).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_empty_payload_store_slot_uses_fixed_zero_send_slot():
    zero_slot = torch.empty(128, dtype=torch.uint8, device="cuda:0")
    payload = torch.empty(0, dtype=torch.uint8, device="cuda:0")

    slot = distributed._payload_store_slot(
        payload,
        128,
        device=torch.device("cuda:0"),
        zero_slot=zero_slot,
    )

    assert slot.data_ptr() == zero_slot.data_ptr()
    assert torch.count_nonzero(slot).item() == 0


def test_distributed_state_from_storage_rejects_cpu_fake_storage(monkeypatch):
    config = RacerConfig(k=1, m=1, train_ranks=(0, 1), spare_ranks=(2,))
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    matrix = cauchy.generate_systematic_matrix(config.k, config.m, config.w)
    group_sizes = {group[0].relative_index: 4 for group in layout.reduction_groups}
    packet_sizes = {0: 4, 1: 0}
    plan = routing.make_planner(config).plan(layout, matrix, 4)
    manifest = distributed._build_storage_manifest(
        tag="dist",
        config=config,
        layout=layout,
        matrix=matrix,
        plan=plan,
        group_nbytes=group_sizes,
        packet_sizes=packet_sizes,
    )

    class FakeStorage:
        def get_manifest(self, tag):
            assert tag == "dist"
            return manifest

    monkeypatch.setattr(distributed, "_require_nccl", lambda process_group=None: None)
    monkeypatch.setattr(distributed, "_rank", lambda process_group=None: 0)
    monkeypatch.setattr(distributed, "_current_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(distributed, "_barrier", lambda process_group=None: None)

    with pytest.raises(RuntimeError, match="CPU/socket fallback get is disabled"):
        distributed.distributed_state_from_storage(
            config=config,
            tag="dist",
            chunk_storage=FakeStorage(),
        )


def test_write_storage_chunks_rejects_cpu_fake_storage(monkeypatch):
    config = RacerConfig(k=3, m=1, train_ranks=(0, 1, 2, 3), spare_ranks=(4,))
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    matrix = cauchy.generate_systematic_matrix(config.k, config.m, config.w)
    group_sizes = {group[0].relative_index: 4 for group in layout.reduction_groups}
    packet_sizes = {0: 4, 1: 4, 2: 4, 3: 4}
    plan = routing.make_planner(config).plan(layout, matrix, 4)
    manifest = distributed._build_storage_manifest(
        tag="dist",
        config=config,
        layout=layout,
        matrix=matrix,
        plan=plan,
        group_nbytes=group_sizes,
        packet_sizes=packet_sizes,
    )
    state = distributed.DistributedStoreResult(
        tag="dist",
        config=config,
        layout=layout,
        matrix=matrix,
        plan=plan,
        local_chunks={
            "rg_000000_row_001": torch.ones(4, dtype=torch.uint8),
            "rg_000001_row_001": torch.zeros(4, dtype=torch.uint8),
        },
        packet_nbytes_by_rank=packet_sizes,
        manifest=manifest,
    )

    class FakeStorage:
        pass

    storage = FakeStorage()
    monkeypatch.setattr(distributed, "_barrier", lambda process_group=None: None)

    with pytest.raises(RuntimeError, match="CPU/socket fallback put is disabled"):
        distributed._write_storage_chunks(
            chunk_storage=storage,
            state=state,
            rank=1,
        )


def test_distributed_store_writes_each_reduction_group_before_next_group(monkeypatch):
    config = RacerConfig(k=3, m=1, train_ranks=(0, 1, 2, 3), spare_ranks=(4,))
    observed_puts: list[tuple[str, ...]] = []
    events: list[str] = []

    def fake_store_data_rows_for_group(**kwargs):
        group = kwargs["group"]
        chunks = kwargs["local_chunks"]
        group_id = int(group[0].relative_index)
        chunks[f"rg_{group_id:06d}_row_000"] = torch.ones(4, dtype=torch.uint8)
        return 0

    def fake_store_parity_for_group(**kwargs):
        group = kwargs["group"]
        chunks = kwargs["local_chunks"]
        group_id = int(group[0].relative_index)
        chunks[f"rg_{group_id:06d}_row_003"] = torch.ones(4, dtype=torch.uint8)
        return 0

    def fake_put_storage_chunks(**kwargs):
        chunks = kwargs["chunks"]
        chunk_ids = tuple(sorted(chunks))
        assert chunk_ids
        group_ids = {chunk_id.split("_row_", 1)[0] for chunk_id in chunk_ids}
        assert len(group_ids) == 1
        observed_puts.append(chunk_ids)
        return sum(int(chunk.numel()) for chunk in chunks.values()), len(chunks), {"storage_wait_ms": 1.0}

    monkeypatch.setattr(distributed, "_require_nccl", lambda process_group=None: None)
    monkeypatch.setattr(distributed, "_rank", lambda process_group=None: 0)
    monkeypatch.setattr(distributed, "_barrier", lambda process_group=None: None)
    monkeypatch.setattr(distributed, "_cuda_payload", lambda local_packet: local_packet)
    monkeypatch.setattr(
        distributed,
        "_payload_store_slot",
        lambda payload, nbytes, *, device, zero_slot=None: payload.narrow(0, 0, int(nbytes)),
    )
    monkeypatch.setattr(distributed, "_begin_storage_checkpoint", lambda **kwargs: events.append("begin") or {})
    monkeypatch.setattr(distributed, "_commit_storage_checkpoint", lambda **kwargs: events.append("commit") or {})
    monkeypatch.setattr(distributed, "_store_data_rows_for_group", fake_store_data_rows_for_group)
    monkeypatch.setattr(distributed, "_store_spare_compute_parity_cuda_for_group", fake_store_parity_for_group)
    monkeypatch.setattr(distributed, "_put_storage_chunks", fake_put_storage_chunks)

    result = distributed.distributed_store(
        config=config,
        local_packet=torch.ones(4, dtype=torch.uint8),
        tag="dist",
        chunk_storage=object(),
        packet_sizes_by_rank={0: 4, 1: 4, 2: 4, 3: 4},
    )

    assert events == ["begin", "commit"]
    assert observed_puts == [
        ("rg_000000_row_000", "rg_000000_row_003"),
        ("rg_000001_row_000", "rg_000001_row_003"),
    ]
    assert result.local_chunks == {}
    assert result.profile["local_chunks_released_count"] == 4
    assert result.profile["local_storage_chunk_count"] == 4


def test_store_data_rows_receives_into_preallocated_slot_and_flushes(monkeypatch):
    config = RacerConfig(k=3, m=1, train_ranks=(0, 1, 2, 3), spare_ranks=(4,))
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    group = layout.reduction_groups[1]
    receive_slot = torch.empty(16, dtype=torch.uint8)
    stored: list[tuple[str, int]] = []

    def fail_recv_tensor(*args, **kwargs):
        raise AssertionError("store path must receive into a fixed slot instead of allocating")

    def fake_recv_into(dst, nbytes, src, *, process_group=None):
        assert dst.data_ptr() == receive_slot.data_ptr()
        return dst.narrow(0, 0, int(nbytes)).fill_(int(src))

    monkeypatch.setattr(distributed, "_current_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(distributed, "_recv_tensor", fail_recv_tensor)
    monkeypatch.setattr(distributed, "_recv_tensor_into", fake_recv_into)

    sent = distributed._store_data_rows_for_group(
        rank=0,
        config=config,
        group=group,
        group_nbytes={group[0].relative_index: 16},
        local_slot_payload={},
        local_chunks={},
        receive_slot=receive_slot,
        row_sink=lambda chunk_id, tensor: stored.append((chunk_id, int(tensor.numel()))),
    )

    assert sent == 0
    assert stored == [("rg_000001_row_000", 16)]


def test_parity_receive_reuses_payload_slot3_after_local_send(monkeypatch):
    config = RacerConfig(k=1, m=1, train_ranks=(0, 1), spare_ranks=(2,))
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    group = layout.reduction_groups[1]
    payload_slot3 = torch.empty(16, dtype=torch.uint8)
    payload_view = payload_slot3.narrow(0, 0, 8).fill_(7)
    stored: list[tuple[str, int, int]] = []

    def fail_recv_tensor(*args, **kwargs):
        raise AssertionError("store path must receive parity into a fixed slot instead of allocating")

    def fake_recv_into(dst, nbytes, src, *, process_group=None):
        assert dst.data_ptr() == payload_slot3.data_ptr()
        return dst.narrow(0, 0, int(nbytes)).fill_(int(src))

    monkeypatch.setattr(distributed, "_current_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        distributed,
        "_payload_store_slot",
        lambda payload, nbytes, *, device, zero_slot=None: payload_slot3.narrow(0, 0, int(nbytes)),
    )
    monkeypatch.setattr(distributed, "_send_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed, "_recv_tensor", fail_recv_tensor)
    monkeypatch.setattr(distributed, "_recv_tensor_into", fake_recv_into)

    sent = distributed._store_spare_compute_parity_cuda_for_group(
        rank=1,
        config=config,
        group=group,
        E=cauchy.generate_systematic_matrix(config.k, config.m, config.w),
        group_nbytes={group[0].relative_index: 16},
        local_slot_payload={1: payload_view},
        local_chunks={},
        receive_slot=payload_slot3,
        row_sink=lambda chunk_id, tensor: stored.append((chunk_id, int(tensor.numel()), int(tensor[0].item()))),
    )

    assert sent == 16
    assert stored == [("rg_000001_row_001", 16, 2)]


def test_distributed_state_from_storage_rejects_cpu_fake_storage_with_zero_rows(monkeypatch):
    config = RacerConfig(k=3, m=1, train_ranks=(0, 1, 2, 3), spare_ranks=(4,))
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    matrix = cauchy.generate_systematic_matrix(config.k, config.m, config.w)
    group_sizes = {group[0].relative_index: 4 for group in layout.reduction_groups}
    packet_sizes = {0: 4, 1: 4, 2: 4, 3: 4}
    plan = routing.make_planner(config).plan(layout, matrix, 4)
    manifest = distributed._build_storage_manifest(
        tag="dist",
        config=config,
        layout=layout,
        matrix=matrix,
        plan=plan,
        group_nbytes=group_sizes,
        packet_sizes=packet_sizes,
    )

    class FakeStorage:
        def get_manifest(self, tag):
            assert tag == "dist"
            return manifest

    storage = FakeStorage()
    monkeypatch.setattr(distributed, "_require_nccl", lambda process_group=None: None)
    monkeypatch.setattr(distributed, "_rank", lambda process_group=None: 1)
    monkeypatch.setattr(distributed, "_current_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(distributed, "_barrier", lambda process_group=None: None)

    with pytest.raises(RuntimeError, match="CPU/socket fallback get is disabled"):
        distributed.distributed_state_from_storage(
            config=config,
            tag="dist",
            chunk_storage=storage,
        )


def test_store_data_rows_skips_virtual_zero_without_allocating_zero_buffer(monkeypatch):
    config = RacerConfig(k=3, m=1, train_ranks=(0, 1, 2, 3), spare_ranks=(4,))
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    layout = replace(layout, reduction_groups=(layout.reduction_groups[1],))
    group_sizes = {group[0].relative_index: 16 for group in layout.reduction_groups}
    local_chunks: dict[str, torch.Tensor] = {}

    def fail_zero_buffer(*args, **kwargs):
        raise AssertionError("virtual zero rows must not allocate real CUDA buffers")

    monkeypatch.setattr(distributed, "_current_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(distributed, "_zero_buffer", fail_zero_buffer)

    sent = distributed._store_data_rows(
        rank=1,
        config=config,
        layout=layout,
        group_nbytes=group_sizes,
        local_slot_payload={},
        local_chunks=local_chunks,
    )

    assert sent == 0
    assert local_chunks == {}


def test_spare_parity_skips_virtual_zero_columns(monkeypatch):
    config = RacerConfig(k=3, m=1, train_ranks=(0, 1, 2, 3), spare_ranks=(4,))
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    group_sizes = {group[0].relative_index: 16 for group in layout.reduction_groups}
    matrix = cauchy.generate_systematic_matrix(config.k, config.m, config.w)
    calls: list[tuple[int, int]] = []

    def fail_zero_buffer(*args, **kwargs):
        raise AssertionError("virtual zero columns must not allocate real CUDA buffers")

    def fake_recv(nbytes, src, *, device=None, process_group=None):
        return torch.full((int(nbytes),), int(src), dtype=torch.uint8)

    def fake_apply(inputs, coeff):
        calls.append((len(inputs), len(coeff[0])))
        return [torch.empty_like(inputs[0])]

    monkeypatch.setattr(distributed, "_current_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(distributed, "_zero_buffer", fail_zero_buffer)
    monkeypatch.setattr(distributed, "_recv_tensor", fake_recv)
    monkeypatch.setattr(distributed, "_send_tensor", lambda *args, **kwargs: None)
    monkeypatch.setattr(distributed.codec_cuda, "apply_matrix_cuda", fake_apply)

    sent = distributed._store_spare_compute_parity_cuda(
        rank=4,
        config=config,
        layout=layout,
        E=matrix,
        group_nbytes=group_sizes,
        local_slot_payload={},
        local_chunks={},
    )

    assert sent == 32
    assert calls[-1] == (1, 1)
