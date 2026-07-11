import pytest
import torch
from dataclasses import replace

from racer import cauchy, distributed, routing
from racer.config import RacerConfig
from racer.layout import ElasticLayout


def _storage_manifest_for_config(config: RacerConfig) -> dict:
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    matrix = cauchy.generate_systematic_matrix(config.k, config.m, config.w)
    group_sizes = {group[0].relative_index: 4 for group in layout.reduction_groups}
    packet_sizes = {int(rank): 4 for rank in config.train_ranks}
    plan = routing.make_planner(config).plan(layout, matrix, 4)
    return distributed._build_storage_manifest(
        tag="dist",
        config=config,
        layout=layout,
        matrix=matrix,
        plan=plan,
        group_nbytes=group_sizes,
        packet_sizes=packet_sizes,
    )


def test_per_node_csd_rank_ranges_and_expected_chunk_counts():
    config = RacerConfig(k=6, m=2, train_ranks=tuple(range(8)), spare_ranks=(8,))
    manifest = _storage_manifest_for_config(config)

    assert distributed._csv_ints("0-3,5,7-6") == [0, 1, 2, 3, 5, 7, 6]

    global_count = distributed._expected_storage_chunk_count(manifest)
    first_node_count = distributed._expected_storage_chunk_count_for_ranks(manifest, {0, 1, 2, 3})
    second_node_count = distributed._expected_storage_chunk_count_for_ranks(manifest, {4, 5, 6, 7})
    spare_node_count = distributed._expected_storage_chunk_count_for_ranks(manifest, {8})

    assert global_count > 0
    assert first_node_count > 0
    assert second_node_count > 0
    assert spare_node_count == 0
    assert first_node_count + second_node_count == global_count


def test_per_node_csd_begin_commit_only_runs_on_local_coordinator(monkeypatch):
    config = RacerConfig(k=6, m=2, train_ranks=tuple(range(8)), spare_ranks=(8,))
    manifest = _storage_manifest_for_config(config)
    state = distributed.DistributedStoreResult(
        tag="dist",
        config=config,
        layout=ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m),
        matrix=cauchy.generate_systematic_matrix(config.k, config.m, config.w),
        plan=routing.make_planner(config).plan(
            ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m),
            cauchy.generate_systematic_matrix(config.k, config.m, config.w),
            4,
        ),
        local_chunks={},
        packet_nbytes_by_rank={int(rank): 4 for rank in config.train_ranks},
        manifest=manifest,
    )

    class FakeStorage:
        def __init__(self):
            self.begin_calls = []
            self.manifests = []
            self.commits = []

        def begin(self, tag, *, manifest_base, expected_chunks):
            self.begin_calls.append((tag, int(expected_chunks)))

        def put_manifest(self, tag, manifest):
            self.manifests.append((tag, manifest))

        def commit(self, tag):
            self.commits.append(tag)

    monkeypatch.setenv("RACER_CSD_PER_NODE", "1")
    monkeypatch.setenv("RACER_CSD_LOCAL_RANKS", "4-7")
    monkeypatch.setenv("RACER_CSD_LOCAL_COORDINATOR_RANK", "4")
    monkeypatch.setattr(distributed, "_barrier", lambda process_group=None: None)

    expected_local_count = distributed._expected_storage_chunk_count_for_ranks(manifest, {4, 5, 6, 7})
    storage = FakeStorage()

    distributed._begin_storage_checkpoint(chunk_storage=storage, state=state, rank=4)
    distributed._commit_storage_checkpoint(chunk_storage=storage, state=state, rank=4)

    assert storage.begin_calls == [("dist", expected_local_count)]
    assert storage.manifests == [("dist", manifest)]
    assert storage.commits == ["dist"]

    storage = FakeStorage()
    distributed._begin_storage_checkpoint(chunk_storage=storage, state=state, rank=5)
    distributed._commit_storage_checkpoint(chunk_storage=storage, state=state, rank=5)
    assert storage.begin_calls == []
    assert storage.manifests == []
    assert storage.commits == []

    storage = FakeStorage()
    distributed._commit_storage_checkpoint(chunk_storage=storage, state=state, rank=0)
    assert storage.commits == []


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
    manifest.update({"committed": True, "daemon_owned": True, "data_resident": True})

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
        assert kwargs["row_sink"] is None
        group = kwargs["group"]
        chunks = kwargs["local_chunks"]
        group_id = int(group[0].relative_index)
        events.append(f"group{group_id}:data_rows")
        chunks[f"rg_{group_id:06d}_row_000"] = torch.ones(4, dtype=torch.uint8)
        return 0

    def fake_store_parity_for_group(**kwargs):
        assert kwargs["row_sink"] is None
        group = kwargs["group"]
        chunks = kwargs["local_chunks"]
        group_id = int(group[0].relative_index)
        events.append(f"group{group_id}:parity")
        chunks[f"rg_{group_id:06d}_row_003"] = torch.ones(4, dtype=torch.uint8)
        return 0

    def fake_put_storage_chunks(**kwargs):
        chunks = kwargs["chunks"]
        chunk_ids = tuple(sorted(chunks))
        assert chunk_ids
        group_ids = {chunk_id.split("_row_", 1)[0] for chunk_id in chunk_ids}
        assert len(group_ids) == 1
        observed_puts.append(chunk_ids)
        events.append(f"{next(iter(group_ids))}:put")
        return sum(int(chunk.numel()) for chunk in chunks.values()), len(chunks), {"storage_wait_ms": 1.0}

    monkeypatch.setattr(distributed, "_require_nccl", lambda process_group=None: None)
    monkeypatch.setattr(distributed, "_rank", lambda process_group=None: 0)
    monkeypatch.setattr(distributed, "_barrier", lambda process_group=None: events.append("barrier"))
    monkeypatch.setattr(
        distributed,
        "_stream_barrier",
        lambda process_group=None: events.append("stream_barrier"),
    )
    monkeypatch.setattr(distributed, "_current_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(distributed, "_cuda_payload", lambda local_packet: local_packet)
    monkeypatch.setattr(
        distributed,
        "_payload_store_slot",
        lambda payload, nbytes, *, device, zero_slot=None: payload,
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

    assert events == [
        "begin",
        "group0:data_rows",
        "group0:parity",
        "barrier",
        "rg_000000:put",
        "barrier",
        "group1:data_rows",
        "group1:parity",
        "barrier",
        "rg_000001:put",
        "barrier",
        "commit",
    ]
    assert observed_puts == [
        ("rg_000000_row_000", "rg_000000_row_003"),
        ("rg_000001_row_000", "rg_000001_row_003"),
    ]
    assert result.local_chunks == {}
    assert result.profile["local_chunks_released_count"] == 4
    assert result.profile["local_storage_chunk_count"] == 4
    assert "group_transfer_barrier_ms" in result.profile
    assert "group_storage_barrier_ms" in result.profile


def test_distributed_store_many_batches_control_barriers_and_preserves_tags(monkeypatch):
    config = RacerConfig(k=3, m=1, train_ranks=(0, 1, 2, 3), spare_ranks=(4,))
    tags = ["checkpoint:chunk:000000", "checkpoint:chunk:000001"]
    events: list[str] = []

    class FakeStorage:
        def begin(self, tag, *, manifest_base, expected_chunks):
            assert manifest_base["tag"] == tag
            assert int(expected_chunks) > 0
            events.append(f"begin:{tag}")

        def put_manifest(self, tag, manifest):
            assert manifest["tag"] == tag
            events.append(f"manifest:{tag}")

        def commit(self, tag):
            events.append(f"commit:{tag}")

        def wait(self, op_id):
            events.append(f"wait:{op_id}")
            return {"profile": {}}

    def fake_store_data_rows_for_group(**kwargs):
        group = kwargs["group"]
        chunks = kwargs["local_chunks"]
        group_id = int(group[0].relative_index)
        events.append(f"group{group_id}:data_rows")
        chunks[f"rg_{group_id:06d}_row_000"] = torch.ones(4, dtype=torch.uint8)
        return 0

    def fake_store_parity_for_group(**kwargs):
        group = kwargs["group"]
        chunks = kwargs["local_chunks"]
        group_id = int(group[0].relative_index)
        events.append(f"group{group_id}:parity")
        chunks[f"rg_{group_id:06d}_row_003"] = torch.ones(4, dtype=torch.uint8)
        return 0

    def fake_enqueue_storage_chunks(**kwargs):
        state = kwargs["state"]
        chunks = kwargs["chunks"]
        chunk_ids = tuple(sorted(chunks))
        group_ids = {chunk_id.split("_row_", 1)[0] for chunk_id in chunk_ids}
        assert len(group_ids) == 1
        group_id = next(iter(group_ids))
        events.append(f"{state.tag}:{group_id}:enqueue")
        pending = [
            distributed.DistributedStorePendingPut(
                tag=state.tag,
                chunk_id=chunk_id,
                op_id=f"{state.tag}:{chunk_id}",
                tensor=chunks[chunk_id],
            )
            for chunk_id in chunk_ids
        ]
        return (
            sum(int(chunk.numel()) for chunk in chunks.values()),
            len(chunks),
            {"storage_enqueue_ms": 1.0},
            pending,
        )

    monkeypatch.setattr(distributed, "_require_nccl", lambda process_group=None: None)
    monkeypatch.setattr(distributed, "_rank", lambda process_group=None: 0)
    monkeypatch.setattr(distributed, "_barrier", lambda process_group=None: events.append("barrier"))
    monkeypatch.setattr(
        distributed,
        "_stream_barrier",
        lambda process_group=None: events.append("stream_barrier"),
    )
    monkeypatch.setattr(distributed, "_current_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(distributed, "_cuda_payload", lambda local_packet: local_packet)
    monkeypatch.setattr(
        distributed,
        "_payload_store_slot",
        lambda payload, nbytes, *, device, zero_slot=None: payload,
    )
    monkeypatch.setattr(distributed, "_store_data_rows_for_group", fake_store_data_rows_for_group)
    monkeypatch.setattr(distributed, "_store_spare_compute_parity_cuda_for_group", fake_store_parity_for_group)
    monkeypatch.setattr(distributed, "_enqueue_storage_chunks", fake_enqueue_storage_chunks)

    packet_sizes = {0: 4, 1: 4, 2: 4, 3: 4}
    storage = FakeStorage()
    handle = distributed.prepare_distributed_store_many(
        config=config,
        tags=tags,
        local_packets=[
            torch.ones(4, dtype=torch.uint8),
            torch.zeros(4, dtype=torch.uint8),
        ],
        chunk_storage=storage,
        packet_sizes_by_tag={tag: packet_sizes for tag in tags},
    )

    expected_prepare_events = [
        f"begin:{tags[0]}",
        f"begin:{tags[1]}",
        "barrier",
        "group0:data_rows",
        "group0:parity",
        "stream_barrier",
        f"{tags[0]}:rg_000000:enqueue",
        "group1:data_rows",
        "group1:parity",
        "stream_barrier",
        f"{tags[0]}:rg_000001:enqueue",
        "group0:data_rows",
        "group0:parity",
        "stream_barrier",
        f"{tags[1]}:rg_000000:enqueue",
        "group1:data_rows",
        "group1:parity",
        "stream_barrier",
        f"{tags[1]}:rg_000001:enqueue",
    ]
    assert events == expected_prepare_events
    assert events.count("barrier") == 1
    assert not any(event.startswith(("wait:", "manifest:", "commit:")) for event in events)
    assert [state.tag for state in handle.states] == tags
    assert all(not state.storage_backed for state in handle.states)
    assert all(len(pending) == 4 for pending in handle.pending_puts_by_tag)
    receive_slots = [
        tensor
        for tensors in handle.retained_tensors_by_tag
        for tensor in tensors
        if int(tensor.numel()) > 4
    ]
    assert len(receive_slots) == 4
    assert len({int(tensor.data_ptr()) for tensor in receive_slots}) == 4

    barrier_count_before_finalize = events.count("barrier")
    states = distributed.finalize_distributed_store_many(handle)

    assert events.count("barrier") == barrier_count_before_finalize
    assert [event for event in events if event.startswith("manifest:")] == [
        f"manifest:{tags[0]}",
        f"manifest:{tags[1]}",
    ]
    assert [event for event in events if event.startswith("commit:")] == [
        f"commit:{tags[0]}",
        f"commit:{tags[1]}",
    ]
    assert len([event for event in events if event.startswith("wait:")]) == 8
    assert [state.tag for state in states] == tags
    assert all(state.storage_backed for state in states)
    assert all(state.local_chunks == {} for state in states)
    assert all(not pending for pending in handle.pending_puts_by_tag)
    assert all(not tensors for tensors in handle.retained_tensors_by_tag)
    assert sum(int(state.profile["store_batch_tag_count"]) for state in states) == 2
    assert all(float(state.profile["group_storage_barrier_ms"]) == 0.0 for state in states)
    assert all(float(state.profile["storage_commit_pre_barrier_ms"]) == 0.0 for state in states)
    assert all(float(state.profile["storage_commit_post_barrier_ms"]) == 0.0 for state in states)
    assert all(int(state.profile["local_storage_chunk_count"]) == 4 for state in states)


def test_stream_barrier_installs_stream_dependency_without_waiting():
    events = []

    class FakeWork:
        def synchronize(self):
            events.append("synchronize")

        def wait(self):
            pytest.fail("stream barrier must not call blocking Work.wait()")

    class FakeProcessGroup:
        def barrier(self):
            events.append("barrier")
            return FakeWork()

    distributed._stream_barrier(FakeProcessGroup())

    assert events == ["barrier", "synchronize"]


def test_csd_only_commit_retries_incomplete_seals_and_rejects_other_errors(monkeypatch):
    config = RacerConfig(k=1, m=1, train_ranks=(0, 1), spare_ranks=(2,))
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    matrix = cauchy.generate_systematic_matrix(config.k, config.m, config.w)
    packet_sizes = {0: 4, 1: 4}
    group_sizes = {group[0].relative_index: 4 for group in layout.reduction_groups}
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
        local_chunks={},
        packet_nbytes_by_rank=packet_sizes,
        manifest=manifest,
    )

    class EventuallySealedStorage:
        def __init__(self):
            self.commit_calls = 0

        def put_manifest(self, tag, value):
            assert tag == "dist"
            assert value is manifest

        def commit(self, tag):
            self.commit_calls += 1
            if self.commit_calls == 1:
                raise RuntimeError("CSD checkpoint 'dist' cannot commit: sealed_chunks=0, expected_chunks=2")
            if self.commit_calls == 2:
                raise RuntimeError("CSD checkpoint 'dist' cannot commit: sealed_chunks=1, expected_chunks=2")

    monkeypatch.delenv("RACER_CSD_PER_NODE", raising=False)
    monkeypatch.setenv("RACER_CSD_COMMIT_WAIT_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("RACER_CSD_COMMIT_RETRY_INTERVAL_MS", "1")
    monkeypatch.setattr(distributed.time, "sleep", lambda _seconds: None)
    storage = EventuallySealedStorage()
    profile = distributed._commit_storage_checkpoint_after_local_waits(
        chunk_storage=storage,
        state=state,
        rank=0,
    )
    assert storage.commit_calls == 3
    assert profile["storage_commit_retry_count"] == 2.0
    assert profile["storage_commit_pre_barrier_ms"] == 0.0
    assert profile["storage_commit_post_barrier_ms"] == 0.0

    storage.commit_calls = 10
    storage.commit = lambda _tag: (_ for _ in ()).throw(RuntimeError("unrelated storage failure"))
    with pytest.raises(RuntimeError, match="unrelated storage failure"):
        distributed._commit_storage_checkpoint_after_local_waits(
            chunk_storage=storage,
            state=state,
            rank=0,
        )


def test_distributed_store_many_validates_batch_shape_before_distributed_setup(monkeypatch):
    config = RacerConfig(k=1, m=1, train_ranks=(0, 1), spare_ranks=(2,))

    assert distributed.distributed_store_many(
        config=config,
        tags=[],
        local_packets=[],
    ) == []
    with pytest.raises(ValueError, match="one local packet per tag"):
        distributed.distributed_store_many(
            config=config,
            tags=["a"],
            local_packets=[],
        )
    with pytest.raises(ValueError, match="unique tags"):
        distributed.distributed_store_many(
            config=config,
            tags=["a", "a"],
            local_packets=[None, None],
        )

    monkeypatch.setattr(distributed, "_require_nccl", lambda process_group=None: None)
    monkeypatch.setattr(distributed, "_rank", lambda process_group=None: 0)
    monkeypatch.setattr(distributed, "_cuda_payload", lambda local_packet: local_packet)
    aliased = torch.ones(4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="must not alias"):
        distributed.distributed_store_many(
            config=config,
            tags=["a", "b"],
            local_packets=[aliased, aliased],
            chunk_storage=object(),
            packet_sizes_by_tag={
                "a": {0: 4, 1: 4},
                "b": {0: 4, 1: 4},
            },
        )


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
    manifest.update({"committed": True, "daemon_owned": True, "data_resident": True})

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


def test_distributed_load_defaults_failed_rank_output_to_first_spare(monkeypatch):
    config = RacerConfig(k=1, m=1, train_ranks=(0, 1), spare_ranks=(2,))
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    matrix = cauchy.generate_systematic_matrix(config.k, config.m, config.w)
    plan = routing.make_planner(config).plan(layout, matrix, 4)
    state = distributed.DistributedStoreResult(
        tag="dist",
        config=config,
        layout=layout,
        matrix=matrix,
        plan=plan,
        local_chunks={},
        packet_nbytes_by_rank={0: 4, 1: 4},
    )

    monkeypatch.setattr(distributed, "_require_nccl", lambda process_group=None: None)
    monkeypatch.setattr(distributed, "_rank", lambda process_group=None: 2)
    monkeypatch.setattr(distributed, "_current_cuda_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(distributed, "_barrier", lambda process_group=None: None)
    monkeypatch.setattr(distributed, "_recv_tensor", lambda nbytes, src, *, device=None, process_group=None: torch.arange(int(nbytes), dtype=torch.uint8))
    monkeypatch.setattr(distributed.codec_cuda, "decode_blocks", lambda chunks, rows, matrix: [chunks[0]])

    result = distributed.distributed_load(
        state=state,
        failed_train_ranks=[0],
        requested_train_ranks=[0],
        replacement_mapping=None,
    )

    assert result.decode_rank == 2
    assert torch.equal(result.recovered[0], torch.arange(4, dtype=torch.uint8))


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
