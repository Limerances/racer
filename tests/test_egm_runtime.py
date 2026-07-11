import ctypes
import subprocess
import sys

import pytest

import racer.csd as csd
import racer.egm_runtime as egm


class _FakeCudaFn:
    def __init__(self, impl):
        self.impl = impl

    def __call__(self, *args):
        return self.impl(*args)


class _FakeCudart:
    def __init__(self):
        self.created_pools = []
        self.pool_access = []
        self.allocations = []
        self.freed_ptrs = []
        self.destroyed_pools = []
        self.destroyed_streams = []
        self._next_pool = 0xC5D0
        self._next_ptr = 0x100000
        self._next_stream = 0x5000
        self.cudaMemPoolCreate = _FakeCudaFn(self._mem_pool_create)
        self.cudaMemPoolSetAccess = _FakeCudaFn(self._mem_pool_set_access)
        self.cudaMallocFromPoolAsync = _FakeCudaFn(self._malloc_from_pool)
        self.cudaFreeAsync = _FakeCudaFn(self._free_async)
        self.cudaMemPoolDestroy = _FakeCudaFn(self._mem_pool_destroy)
        self.cudaStreamCreateWithFlags = _FakeCudaFn(self._stream_create)
        self.cudaStreamSynchronize = _FakeCudaFn(lambda *_args: 0)
        self.cudaStreamDestroy = _FakeCudaFn(self._stream_destroy)
        self.cudaIpcCloseMemHandle = _FakeCudaFn(lambda *_args: 0)
        self.cudaEventRecord = _FakeCudaFn(lambda *_args: 0)
        self.cudaStreamWaitEvent = _FakeCudaFn(lambda *_args: 0)
        self.cudaMemcpyAsync = _FakeCudaFn(lambda *_args: 0)

    @staticmethod
    def _value(value):
        return int(value.value if hasattr(value, "value") else value)

    def _mem_pool_create(self, pool_ptr, props_ptr):
        pool = self._next_pool
        self._next_pool += 1
        props = props_ptr._obj
        self.created_pools.append(
            {
                "pool": pool,
                "alloc_type": int(props.allocType),
                "location_type": int(props.location.type),
                "location_id": int(props.location.id),
                "max_size": int(props.maxSize),
            }
        )
        pool_ptr._obj.value = pool
        return 0

    def _mem_pool_set_access(self, pool, descs, count):
        self.pool_access.append(
            {
                "pool": self._value(pool),
                "devices": [int(descs[index].location.id) for index in range(self._value(count))],
            }
        )
        return 0

    def _malloc_from_pool(self, ptr, size, pool, stream):
        nbytes = self._value(size)
        address = self._next_ptr
        self._next_ptr += max(4096, nbytes)
        ptr._obj.value = address
        self.allocations.append(
            {
                "ptr": address,
                "nbytes": nbytes,
                "pool": self._value(pool),
                "stream": self._value(stream),
            }
        )
        return 0

    def _free_async(self, ptr, stream):
        self.freed_ptrs.append((self._value(ptr), self._value(stream)))
        return 0

    def _mem_pool_destroy(self, pool):
        self.destroyed_pools.append(self._value(pool))
        return 0

    def _stream_create(self, stream_ptr, _flags):
        stream_ptr._obj.value = self._next_stream
        self._next_stream += 1
        return 0

    def _stream_destroy(self, stream):
        self.destroyed_streams.append(self._value(stream))
        return 0


def _patch_cuda(monkeypatch, device_numa):
    fake = _FakeCudart()
    monkeypatch.setenv("RACER_CSD_PREWARM_CUDA_CONTEXTS", "0")
    monkeypatch.setenv("RACER_CSD_COPY_STREAMS_PER_DEVICE", "1")
    monkeypatch.setattr(egm, "_configure_cudart_mempool_api", lambda: fake)
    monkeypatch.setattr(egm, "_load_cudart", lambda: fake)
    monkeypatch.setattr(egm, "_detect_host_numa_id", lambda device: int(device_numa[int(device)]))
    monkeypatch.setattr(egm, "_visible_cuda_device_count", lambda: len(device_numa))
    monkeypatch.setattr(csd, "_load_cudart", lambda: fake)
    monkeypatch.setattr(csd, "_cuda_set_device", lambda device: None)
    monkeypatch.setattr(csd.torch.cuda, "is_available", lambda: True)
    return fake


def test_builtin_egm_runtime_uses_per_device_host_numa_pools(monkeypatch):
    fake = _patch_cuda(monkeypatch, {0: 7, 1: 7, 2: 7, 3: 7})

    runtime = egm.create_runtime(home_device=2, total_bytes=0, segment_bytes=4096)

    assert len(fake.created_pools) == 4
    assert all(pool["alloc_type"] == 1 for pool in fake.created_pools)
    assert all(pool["location_type"] == 3 for pool in fake.created_pools)
    assert all(pool["location_id"] == 7 for pool in fake.created_pools)
    assert runtime.numa_id == 7
    assert runtime.accessing_devices == (0, 1, 2, 3)
    caps = runtime.capabilities()
    assert caps["supports_egm_native_transport"] is True
    assert caps["uses_cuda_mempool"] is True
    assert caps["uses_cudaHostAlloc"] is False
    assert caps["topology_aware"] is True
    assert caps["pool_count"] == 4
    assert runtime.dynamic_allocation_source == "dynamic_cuda_mempool"
    runtime.__del__()


def test_egm_runtime_discovers_device_numa_and_splits_global_cap(monkeypatch):
    fake = _patch_cuda(monkeypatch, {0: 0, 1: 0, 2: 1, 3: 1})

    runtime = egm.create_runtime(
        home_device=0,
        total_bytes=0,
        segment_bytes=4096,
        max_pool_bytes=16384,
    )

    caps = runtime.capabilities()
    assert caps["device_numa_map"] == {"0": 0, "1": 0, "2": 1, "3": 1}
    assert caps["numa_pool_count"] == 2
    assert caps["pool_count"] == 4
    assert [pool["max_size"] for pool in fake.created_pools] == [4096, 4096, 4096, 4096]
    assert len(fake.pool_access) == 4
    assert all(item["devices"] == [0, 1, 2, 3] for item in fake.pool_access)
    runtime.__del__()


def test_scalar_numa_override_only_applies_to_home_device(monkeypatch):
    _patch_cuda(monkeypatch, {0: 0, 1: 0, 2: 1, 3: 1})

    runtime = egm.create_runtime(
        home_device=0,
        numa_id=9,
        total_bytes=0,
        segment_bytes=4096,
    )

    assert runtime.capabilities()["device_numa_map"] == {"0": 9, "1": 0, "2": 1, "3": 1}
    runtime.__del__()


def test_egm_runtime_preallocates_fairly_and_reuses_only_local_device(monkeypatch):
    fake = _patch_cuda(monkeypatch, {0: 0, 1: 0, 2: 1, 3: 1})
    runtime = egm.create_runtime(
        total_bytes=16384,
        segment_bytes=2048,
        max_pool_bytes=16384,
    )

    caps = runtime.capabilities()
    assert caps["preallocated_total_bytes"] == 16384
    assert caps["pool_total_bytes"] == 16384
    assert caps["pool_segment_count"] == 8
    for device in range(4):
        stats = caps["device_pool_stats"][str(device)]
        assert stats["preallocated_bytes"] == 4096
        assert stats["pool_total_bytes"] == 4096
        assert stats["pool_segment_count"] == 2

    first = runtime._allocate_location(1024, allocation_device=0)
    assert first.segment.allocation_device == 0
    assert first.segment.numa_id == 0
    with runtime._lock:
        runtime._free_location_locked(first.segment.segment_id, first.offset, first.nbytes)
    reused = runtime._allocate_location(1024, allocation_device=0)
    other_device = runtime._allocate_location(1024, allocation_device=1)
    assert reused.source == "free_list"
    assert reused.segment.segment_id == first.segment.segment_id
    assert reused.offset == first.offset
    assert other_device.segment.segment_id != first.segment.segment_id

    with pytest.raises(RuntimeError, match="not configured for topology-aware EGM"):
        runtime._allocate_location(1024, allocation_device=7)

    runtime.__del__()
    assert len(fake.freed_ptrs) == 8
    assert len(fake.destroyed_pools) == 4


def test_egm_write_routes_by_ipc_device_and_preserves_placement_metadata(monkeypatch):
    _patch_cuda(monkeypatch, {0: 0, 1: 0, 2: 1, 3: 1})
    runtime = egm.create_runtime(
        total_bytes=16384,
        segment_bytes=4096,
        max_pool_bytes=16384,
    )
    monkeypatch.setattr(
        runtime,
        "_open_ipc_mem",
        lambda view: (0x200000, 0x3000, "", 1.0, 2.0, True, 0.1),
    )
    monkeypatch.setattr(
        runtime,
        "_new_stream_and_events",
        lambda device: (0x4000 + int(device), 0x5000, 0x5001, 0x5002, False),
    )

    op_id, metadata = runtime.write_from_cuda_ipc(
        "tag",
        "chunk",
        {"device": 3, "nbytes": 1024, "base_offset": 0},
        {"checksum_type": "none"},
    )

    assert metadata["stored_device"] == "cuda:3"
    assert metadata["daemon_storage_device"] == "host_numa:1"
    assert metadata["source_device"] == 3
    assert metadata["egm_home_device"] == 3
    assert metadata["egm_numa_id"] == 1
    assert metadata["location"]["pool_id"] == "egm_device_3_numa_1"
    profile = runtime._ops[op_id].profile
    assert profile["daemon_allocation_device"] == 3
    assert profile["daemon_allocation_numa_id"] == 1

    wrapper = csd.EgmBackend(
        runtime=runtime,
        home_device=0,
        numa_id=0,
        accessing_devices=[0, 1, 2, 3],
    )
    normalized = wrapper._normalize_metadata("tag", "chunk", 1024, {}, metadata)
    assert normalized["stored_device"] == "cuda:3"
    assert normalized["daemon_storage_device"] == "host_numa:1"
    assert normalized["egm_home_device"] == 3
    assert normalized["egm_numa_id"] == 1
    assert normalized["location"]["source_device"] == 3
    assert normalized["location"]["numa_id"] == 1
    runtime.__del__()


def test_racer_package_lazily_exports_csd_symbols():
    code = (
        "import sys, racer; "
        "print('before', 'racer.csd' in sys.modules); "
        "print(racer.CheckpointStorageDaemonClient.__name__); "
        "print('after', 'racer.csd' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=15,
    )

    assert "before False" in result.stdout
    assert "CheckpointStorageDaemonClient" in result.stdout
    assert "after True" in result.stdout


def test_csd_module_entrypoint_does_not_preimport_itself():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "racer.csd",
            "--backend",
            "egm",
            "--egm-runtime-factory",
            "no.such:factory",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=15,
    )

    assert result.returncode != 0
    assert "RuntimeWarning" not in result.stdout
    assert "No module named 'no'" in result.stdout
