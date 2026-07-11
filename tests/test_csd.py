import ctypes
import os
import subprocess
import sys
import threading
import types

import pytest
import torch

import racer
import racer.csd as csd_mod
from racer.csd import CheckpointStorageDaemonClient, EgmBackend, NativePinnedMemoryBackend


def test_csd_rejects_fd_mmap_backend_selection():
    with pytest.raises(ValueError, match="unsupported CSD backend"):
        csd_mod._backend_from_name("fd_mmap_host")


def test_csd_client_rejects_socket_bytes_and_fd_compat_paths():
    client = CheckpointStorageDaemonClient("unused.sock", authkey="racer-csd")
    payload = torch.arange(16, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="fd/mmap client mappings are disabled"):
        CheckpointStorageDaemonClient("unused.sock", authkey="racer-csd", cuda_register_fd_mappings=True)
    with pytest.raises(RuntimeError, match="put_chunk is disabled"):
        client.put_chunk("tag", "c0", payload)
    with pytest.raises(RuntimeError, match="put is disabled"):
        client.put("tag", "c0", payload)
    with pytest.raises(RuntimeError, match="get_chunk is disabled"):
        client.get_chunk("tag", "c0")
    with pytest.raises(RuntimeError, match="get is disabled"):
        client.get("tag", "c0")


def test_csd_accept_loop_ignores_reset_during_auth(monkeypatch):
    class FakeBackend:
        name = "fake"

        def capabilities(self):
            return {
                "backend": "fake",
                "restart_aware": True,
                "cuda_native_pinned": False,
            }

    class FakeConn:
        def __init__(self):
            self.sent = []

        def recv(self):
            return {"op": "shutdown"}

        def send(self, payload):
            self.sent.append(payload)

        def close(self):
            pass

    listeners = []

    class FakeListener:
        def __init__(self, address, authkey=None):
            self.address = address
            self.accept_calls = 0
            self.closed = threading.Event()
            listeners.append(self)

        def accept(self):
            self.accept_calls += 1
            if self.accept_calls == 1:
                raise ConnectionResetError("client reset during auth")
            if self.accept_calls == 2:
                return FakeConn()
            self.closed.wait(timeout=2.0)
            raise OSError("listener closed")

        def close(self):
            self.closed.set()

    monkeypatch.setattr(csd_mod, "_backend_from_name", lambda name, options: FakeBackend())
    monkeypatch.setattr(csd_mod, "_visible_cuda_device_count", lambda: 0)
    monkeypatch.setattr(csd_mod, "Listener", FakeListener)

    csd_mod._serve(("fake", 0), b"racer-csd", "fake", {}, metadata_dir=None)

    assert listeners
    assert listeners[0].accept_calls >= 2


def test_native_pinned_backend_reuses_freed_pool_blocks(monkeypatch):
    class FakeCudart:
        def __init__(self):
            self.buffers = []

        def cudaHostAlloc(self, ptr_ref, size, flags):
            nbytes = int(size.value if hasattr(size, "value") else size)
            buffer = ctypes.create_string_buffer(nbytes)
            self.buffers.append(buffer)
            ctypes.cast(ptr_ref, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(ctypes.addressof(buffer))
            return 0

        def cudaFreeHost(self, ptr):
            return 0

    fake = FakeCudart()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: fake)
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda device: None)

    backend = NativePinnedMemoryBackend(total_bytes=1024, segment_bytes=1024)
    assert len(fake.buffers) == 1

    with pytest.raises(NotImplementedError, match="forbids CPU byte writes"):
        backend.write_from_bytes("tag1", "forbidden", b"a")

    backend.allocate("tag1", "c0", 256, {"checksum_type": "none"})
    first = backend.metadata("tag1", "c0")
    assert first["allocation_source"] == "pool_bump"

    backend.allocate("tag1", "c1", 256, {"checksum_type": "none"})
    backend.free("tag1", "c0")
    backend.allocate("tag2", "c0", 128, {"checksum_type": "none"})
    reused = backend.metadata("tag2", "c0")

    assert reused["allocation_source"] == "free_list"
    assert reused["segment_id"] == first["segment_id"]
    assert reused["offset"] == first["offset"]
    assert len(fake.buffers) == 1

    backend.allocate("tag3", "big", 2048, {"checksum_type": "none"})
    expanded = backend.metadata("tag3", "big")
    assert expanded["allocation_source"] == "dynamic_cudaHostAlloc"
    assert len(fake.buffers) == 2


def test_native_pinned_read_rejects_undersized_cuda_ipc_view(monkeypatch):
    monkeypatch.setenv("RACER_CSD_PREWARM_CUDA_CONTEXTS", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: object())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda device: None)

    backend = NativePinnedMemoryBackend(total_bytes=0, segment_bytes=1024)
    backend._chunks = {
        "tag": {
            "c0": csd_mod.BackendChunk(
                tensor=torch.empty(0, dtype=torch.uint8),
                metadata={"nbytes": 128},
                nbytes=128,
                capacity_nbytes=128,
                host_ptr=1234,
                segment_id="seg0",
                offset=0,
            )
        }
    }

    with pytest.raises(ValueError, match="too small"):
        backend.read_to_cuda_ipc("tag", "c0", {"nbytes": 127})


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_native_pinned_checkpoint_survives_training_context_restart(tmp_path):
    daemon = racer.start_checkpoint_storage_daemon(
        metadata_dir=tmp_path / "metadata",
        backend="native_pinned",
        backend_options={"segment_bytes": 8 * 1024 * 1024, "device": 0},
    )
    try:
        store_program = f"""
import torch
import racer
from racer.csd import CheckpointStorageDaemonClient

client = CheckpointStorageDaemonClient({daemon.address!r}, authkey={daemon.authkey!r})
ctx = racer.init(
    k=3,
    m=1,
    train_ranks=[0, 1, 2, 3],
    spare_ranks=[4],
    storage_backend="csd_native_pinned",
    storage_options={{"client": client}},
)
obj = {{
    rank: torch.arange(rank * 23, rank * 23 + 4096, dtype=torch.uint8, device=f"cuda:{{rank}}")
    for rank in [0, 1, 2, 3]
}}
racer.store(obj, tag="restartable", context=ctx)
"""
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
        subprocess.run(
            [sys.executable, "-c", store_program],
            cwd=os.getcwd(),
            env=env,
            check=True,
        )

        manifest = daemon.client.get_manifest("restartable")
        assert manifest["committed"] is True
        assert manifest["daemon_owned"] is True
        assert manifest["data_resident"] is True
        assert len(daemon.client.list_chunks("restartable")) == len(manifest["chunks"])

        # Simulate a training process restart: use a new client and a new
        # RacerContext with an empty checkpoint index, while the daemon keeps
        # owning the committed chunks.
        client2 = CheckpointStorageDaemonClient(daemon.address, authkey=daemon.authkey)
        ctx2 = racer.init(
            k=3,
            m=1,
            train_ranks=[0, 1, 2, 3],
            spare_ranks=[4],
            storage_backend="csd_native_pinned",
            storage_options={"client": client2},
        )
        recovered = racer.load(
            tag="restartable",
            failed_train_ranks=[0],
            replacement_mapping={0: 4},
            context=ctx2,
        )

        expected_rank0 = torch.arange(4096, dtype=torch.uint8, device="cuda:0")
        assert recovered[0].device.index == 4
        assert torch.equal(recovered[0].to(expected_rank0.device), expected_rank0)
    finally:
        daemon.shutdown()


def test_egm_backend_requires_daemon_owned_native_pool():
    with pytest.raises(NotImplementedError, match="real daemon-owned EGM runtime"):
        EgmBackend()


def test_egm_backend_rejects_mempool_without_native_transport():
    with pytest.raises(NotImplementedError, match="native daemon-owned EGM runtime"):
        EgmBackend(mem_pool=object())


def test_egm_runtime_factory_can_be_loaded_from_module_spec(monkeypatch):
    class FakeEgmRuntime:
        def __init__(self, runtime_name):
            self.runtime_name = runtime_name

        def capabilities(self):
            return {"runtime_name": self.runtime_name}

        def write_from_cuda_ipc(self, tag, chunk_id, view, metadata):
            return "put-op", {"nbytes": int(view["nbytes"]), "checksum_type": "none"}

        def read_to_cuda_ipc(self, tag, chunk_id, view):
            return "get-op"

    module = types.ModuleType("fake_egm_runtime_module")

    def make_runtime(runtime_name):
        return FakeEgmRuntime(runtime_name)

    module.make_runtime = make_runtime
    monkeypatch.setitem(sys.modules, "fake_egm_runtime_module", module)

    runtime = csd_mod._load_egm_runtime(
        "fake_egm_runtime_module:make_runtime",
        {"runtime_name": "gb200-test"},
    )
    backend = csd_mod._backend_from_name("egm", {"runtime": runtime, "pool_id": "pool0"})

    caps = backend.capabilities()
    assert caps["backend"] == "egm"
    assert caps["runtime_name"] == "gb200-test"
    assert caps["pool_id"] == "pool0"


def test_metadata_only_csd_manifest_is_resident_when_expected_chunks_zero(tmp_path):
    class MetadataOnlyBackend:
        name = "fake"

        def capabilities(self):
            return {
                "backend": self.name,
                "restart_aware": True,
                "daemon_owned": True,
                "supports_cuda_ipc": True,
                "supports_async_copy": True,
            }

        def list_chunks(self, tag):
            return []

    daemon = csd_mod.CheckpointStorageDaemon(MetadataOnlyBackend(), metadata_dir=tmp_path)
    manifest = {
        "tag": "dist",
        "k": 6,
        "m": 2,
        "train_ranks": list(range(8)),
        "spare_ranks": [8],
        "chunks": [{"chunk_id": "rg_000000_row_000", "owner_rank": 0}],
    }

    daemon.begin("dist", manifest, expected_chunks=0)
    daemon.put_manifest("dist", manifest)
    daemon.commit("dist")
    loaded = daemon.get_manifest("dist")

    assert loaded["committed"] is True
    assert loaded["daemon_owned"] is True
    assert loaded["data_resident"] is True
    assert loaded["expected_chunks"] == 0


def test_egm_backend_delegates_to_native_transport_runtime():
    class FakeEgmRuntime:
        def __init__(self):
            self.metadata_by_tag = {}
            self.waited = []

        def capabilities(self):
            return {"supports_zero_copy_region": True, "runtime": "fake"}

        def write_from_cuda_ipc(self, tag, chunk_id, view, metadata):
            record = {
                "op_id": "put-op",
                "pool_id": "pool0",
                "allocation_id": "alloc0",
                "offset": 4096,
                "nbytes": int(view["nbytes"]),
                "owner_node": "node0",
                "owner_tray": "tray0",
                "access_handle": "handle0",
            }
            self.metadata_by_tag.setdefault(tag, {})[chunk_id] = dict(record)
            return "put-op", record

        def read_to_cuda_ipc(self, tag, chunk_id, view):
            return "get-op"

        def wait(self, op_id):
            self.waited.append(op_id)

        def list_chunks(self, tag):
            return sorted(self.metadata_by_tag.get(tag, {}))

        def metadata(self, tag, chunk_id):
            return dict(self.metadata_by_tag[tag][chunk_id])

    backend = EgmBackend(runtime=FakeEgmRuntime(), pool_id="pool0", owner_node="node0", owner_tray="tray0")

    caps = backend.capabilities()
    assert caps["backend"] == "egm"
    assert caps["supports_egm_native_transport"] is True
    assert caps["supports_cuda_ipc"] is True

    op_id, metadata = backend.write_from_cuda_ipc("tag", "c0", {"nbytes": 123}, {"owner_rank": 0})
    assert op_id == "put-op"
    assert metadata["storage_transport"] == "egm_native"
    assert metadata["egm_allocation_id"] == "alloc0"
    assert metadata["location"]["offset"] == 4096
    assert backend.list_chunks("tag") == ["c0"]
    assert backend.read_to_cuda_ipc("tag", "c0", {"nbytes": 123}) == "get-op"

    with pytest.raises(NotImplementedError, match="CPU/socket byte put"):
        backend.write_from_bytes("tag", "c1", b"abc")


def test_egm_put_wait_does_not_checksum_via_cpu_byte_read(monkeypatch):
    monkeypatch.setattr(csd_mod, "_visible_cuda_device_count", lambda: 0)

    class FakeEgmRuntime:
        def capabilities(self):
            return {"supports_zero_copy_region": True, "runtime": "fake"}

        def write_from_cuda_ipc(self, tag, chunk_id, view, metadata):
            return "put-op", {"nbytes": int(view["nbytes"])}

        def read_to_cuda_ipc(self, tag, chunk_id, view):
            return "get-op"

        def wait(self, op_id):
            pass

    backend = EgmBackend(runtime=FakeEgmRuntime())
    daemon = csd_mod.CheckpointStorageDaemon(backend)
    daemon._entries["tag"] = {
        "tag": "tag",
        "manifest": {},
        "chunks": {},
        "committed": False,
        "data_resident": True,
        "storage_backend": "egm",
    }
    backend.write_from_cuda_ipc("tag", "c0", {"nbytes": 16}, {})
    daemon._async_ops["op0"] = {
        "backend_op_id": "put-op",
        "op_type": "PUT",
        "tag": "tag",
        "chunk_id": "c0",
        "metadata": {},
        "profile": {},
    }

    result = daemon.wait("op0")

    assert result["state"] == "DONE"
    assert "op0" not in daemon._async_ops
    chunk = daemon._entries["tag"]["chunks"]["c0"]
    assert chunk["checksum_type"] == "none"
    assert chunk["checksum"] == ""


def test_native_backend_profile_pops_completed_op():
    backend = NativePinnedMemoryBackend.__new__(NativePinnedMemoryBackend)
    backend._lock = threading.RLock()
    backend._op_condition = threading.Condition(backend._lock)
    backend._ops = {
        "op0": csd_mod._NativeCopyOp(
            op_id="op0",
            device=0,
            stream=0,
            wait_start_event=1,
            copy_start_event=2,
            complete_event=3,
            remote_ptr=None,
            remote_event=None,
            profile={"daemon_memcpy_ms_wall": 1.25},
            done=True,
            resources_released=True,
        )
    }

    assert backend.profile("op0") == {"daemon_memcpy_ms_wall": 1.25}
    assert "op0" not in backend._ops


def test_native_backend_poll_skips_op_claimed_by_wait():
    backend = NativePinnedMemoryBackend.__new__(NativePinnedMemoryBackend)
    backend._lock = threading.RLock()
    backend._op_condition = threading.Condition(backend._lock)
    backend._ops = {
        "op0": csd_mod._NativeCopyOp(
            op_id="op0",
            device=0,
            stream=0,
            wait_start_event=1,
            copy_start_event=2,
            complete_event=3,
            remote_ptr=None,
            remote_event=None,
            in_completion=True,
        )
    }

    assert backend.poll("op0") == "RUNNING"


def test_native_ipc_cache_refcount_decrements_and_evicts_idle(monkeypatch):
    closed_ptrs = []

    class FakeCudaRuntime:
        def cudaIpcCloseMemHandle(self, ptr):
            closed_ptrs.append(int(ptr.value))
            return 0

    backend = NativePinnedMemoryBackend.__new__(NativePinnedMemoryBackend)
    backend._ipc_lock = threading.RLock()
    backend._ipc_mem_cache_max_entries = 1
    backend._ipc_mem_cache = {
        "old": {"ptr": 11, "refcount": 0, "device": 0, "keep_idle": True, "last_used_ns": 1},
        "active": {"ptr": 22, "refcount": 1, "device": 0, "keep_idle": True, "last_used_ns": 2},
    }
    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())

    backend._release_ipc_cache_ref("active")

    assert backend._ipc_mem_cache["active"]["refcount"] == 0
    assert "old" not in backend._ipc_mem_cache
    assert closed_ptrs == [11]


def test_native_direct_ipc_mapping_is_shared_until_last_active_op(monkeypatch):
    opened_ptrs = []
    closed_ptrs = []
    opened_events = []

    class FakeCudaRuntime:
        def cudaIpcOpenMemHandle(self, out_ptr, _handle, _flags):
            out_ptr._obj.value = 1234
            opened_ptrs.append(1234)
            return 0

        def cudaIpcCloseMemHandle(self, ptr):
            closed_ptrs.append(int(ptr.value))
            return 0

        def cudaIpcOpenEventHandle(self, out_event, _handle):
            event = 2000 + len(opened_events)
            out_event._obj.value = event
            opened_events.append(event)
            return 0

    backend = NativePinnedMemoryBackend.__new__(NativePinnedMemoryBackend)
    backend._ipc_lock = threading.RLock()
    backend._ipc_mem_cache_max_entries = 128
    backend._ipc_mem_cache = {}
    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda _device: None)
    view = {
        "device": 0,
        "mem_handle": b"m" * 64,
        "event_handle": b"e" * 64,
        "requires_staging": False,
        "staging_id": None,
        "producer_pid": 99,
        "producer_token": "99:test-session",
    }

    first = backend._open_ipc_mem(view)
    second = backend._open_ipc_mem(view)
    cache_key = first[2]

    assert first[0] == second[0] == 1234
    assert first[5] is True
    assert second[5] is False
    assert opened_ptrs == [1234]
    assert opened_events == [2000, 2001]
    assert backend._ipc_mem_cache[cache_key]["refcount"] == 2

    backend._release_ipc_cache_ref(cache_key)
    assert backend._ipc_mem_cache[cache_key]["refcount"] == 1
    assert closed_ptrs == []

    backend._release_ipc_cache_ref(cache_key)
    assert cache_key not in backend._ipc_mem_cache
    assert closed_ptrs == [1234]


def test_native_direct_source_mapping_reaps_only_after_producer_exit(monkeypatch):
    next_ptr = 3000
    closed_ptrs = []

    class FakeCudaRuntime:
        def cudaIpcOpenMemHandle(self, out_ptr, _handle, _flags):
            nonlocal next_ptr
            out_ptr._obj.value = next_ptr
            next_ptr += 1
            return 0

        def cudaIpcCloseMemHandle(self, ptr):
            closed_ptrs.append(int(ptr.value))
            return 0

        def cudaIpcOpenEventHandle(self, out_event, _handle):
            out_event._obj.value = 4000
            return 0

    backend = NativePinnedMemoryBackend.__new__(NativePinnedMemoryBackend)
    backend._ipc_lock = threading.RLock()
    backend._ipc_mem_cache_max_entries = 128
    backend._ipc_mem_cache = {}
    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda _device: None)
    monkeypatch.setattr(backend, "_producer_pid_is_alive", lambda pid: int(pid) == 101)

    first_view = {
        "device": 0,
        "mem_handle": b"a" * 64,
        "event_handle": b"e" * 64,
        "requires_staging": False,
        "staging_id": None,
        "producer_pid": 100,
        "producer_token": "100:first-session",
        "profile": {"direct_ipc_role": "source"},
    }
    first = backend._open_ipc_mem(first_view)
    backend._release_ipc_cache_ref(first[2])

    assert backend._ipc_mem_cache[first[2]]["refcount"] == 0
    assert closed_ptrs == []

    second_view = dict(first_view)
    second_view.update(
        {
            "mem_handle": b"b" * 64,
            "producer_pid": 101,
            "producer_token": "101:second-session",
        }
    )
    second = backend._open_ipc_mem(second_view)

    assert first[2] not in backend._ipc_mem_cache
    assert backend._ipc_mem_cache[second[2]]["refcount"] == 1
    assert closed_ptrs == [3000]

    backend._release_ipc_cache_ref(second[2])
    assert backend._ipc_mem_cache[second[2]]["refcount"] == 0
    assert closed_ptrs == [3000]


def test_native_source_cache_does_not_alias_distinct_opaque_handles(monkeypatch):
    next_ptr = 5000

    class FakeCudaRuntime:
        def cudaIpcOpenMemHandle(self, out_ptr, _handle, _flags):
            nonlocal next_ptr
            out_ptr._obj.value = next_ptr
            next_ptr += 1
            return 0

        def cudaIpcCloseMemHandle(self, _ptr):
            return 0

        def cudaIpcOpenEventHandle(self, out_event, _handle):
            out_event._obj.value = 6000
            return 0

    backend = NativePinnedMemoryBackend.__new__(NativePinnedMemoryBackend)
    backend._ipc_lock = threading.RLock()
    backend._ipc_mem_cache_max_entries = 128
    backend._ipc_mem_cache = {}
    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda _device: None)
    monkeypatch.setattr(backend, "_producer_pid_is_alive", lambda _pid: True)
    base_view = {
        "device": 0,
        "event_handle": b"e" * 64,
        "requires_staging": False,
        "staging_id": None,
        "producer_pid": 200,
        "producer_token": "200:one-session",
        "profile": {"direct_ipc_role": "source"},
    }
    first_view = dict(base_view, mem_handle=b"x" * 64)
    second_view = dict(base_view, mem_handle=b"y" * 64)

    first = backend._open_ipc_mem(first_view)
    second = backend._open_ipc_mem(second_view)

    assert first[2] != second[2]
    assert first[0] == 5000
    assert second[0] == 5001


def test_csd_allows_explicit_repair_put_after_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(csd_mod, "_visible_cuda_device_count", lambda: 0)

    class FakeCudaBackend:
        name = "fake_cuda"

        def __init__(self):
            self.metadata_by_tag = {}
            self.waited = []

        def capabilities(self):
            return {
                "backend": self.name,
                "restart_aware": True,
                "daemon_owned": True,
                "supports_cuda_ipc": True,
            }

        def write_from_cuda_ipc(self, tag, chunk_id, view, metadata):
            record = dict(metadata)
            record.update(
                {
                    "storage_backend": self.name,
                    "location": {"backend": self.name, "offset": len(self.waited), "nbytes": int(view["nbytes"])},
                    "nbytes": int(view["nbytes"]),
                    "valid_nbytes": int(view["nbytes"]),
                }
            )
            self.metadata_by_tag.setdefault(tag, {})[chunk_id] = record
            return f"backend-{len(self.waited)}", record

        def wait(self, op_id):
            self.waited.append(op_id)

        def profile(self, op_id):
            return {}

        def metadata(self, tag, chunk_id):
            return dict(self.metadata_by_tag[tag][chunk_id])

        def update_metadata(self, tag, chunk_id, metadata):
            self.metadata_by_tag[tag][chunk_id] = dict(metadata)

        def checksum(self, tag, chunk_id, checksum_type="sha256"):
            return f"{checksum_type}-sealed-{len(self.waited)}"

        def list_chunks(self, tag):
            return sorted(self.metadata_by_tag.get(tag, {}))

        def free_tag(self, tag):
            self.metadata_by_tag.pop(tag, None)

    daemon = csd_mod.CheckpointStorageDaemon(FakeCudaBackend(), metadata_dir=tmp_path)
    manifest = {
        "tag": "repairable",
        "k": 1,
        "m": 0,
        "chunks": [{"chunk_id": "c0", "row": 0, "owner_rank": 0, "checksum": "stale"}],
    }
    daemon.begin("repairable", manifest, expected_chunks=1)
    put = daemon.put_cuda_ipc(
        "repairable",
        "c0",
        {"nbytes": 4},
        {"row": 0, "owner_rank": 0, "writer_rank": 0, "nbytes": 4, "checksum_type": "sample64"},
    )
    assert daemon.wait(put["op_id"])["state"] == "DONE"
    daemon.put_manifest("repairable", manifest)
    daemon.commit("repairable")

    repair_manifest = {
        "tag": "repairable",
        "k": 1,
        "m": 0,
        "chunks": [
            {
                "chunk_id": "c0",
                "row": 0,
                "owner_rank": 4,
                "checksum_type": "sample64",
                "checksum": "repair-stale",
            }
        ],
    }
    repair_put = daemon.put_cuda_ipc(
        "repairable",
        "c0",
        {"nbytes": 4},
        {
            "row": 0,
            "owner_rank": 4,
            "writer_rank": 4,
            "nbytes": 4,
            "checksum_type": "sample64",
            "committed_checkpoint_update": True,
        },
    )
    assert daemon.wait(repair_put["op_id"])["state"] == "DONE"
    daemon.put_manifest("repairable", repair_manifest)

    updated = daemon.get_manifest("repairable")
    assert updated["chunks"][0]["owner_rank"] == 4
    assert updated["chunks"][0]["checksum"] != "repair-stale"
    assert updated["chunks"][0]["checksum"].startswith("sample64-sealed")
