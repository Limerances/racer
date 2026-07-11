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
    backend.close()


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
    backend.close()


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


def test_csd_cli_does_not_override_env_or_inject_defaults_into_custom_egm_factory(monkeypatch):
    module = types.ModuleType("zero_arg_egm_runtime_module")
    runtime = object()
    calls = []

    def make_runtime():
        calls.append("called")
        return runtime

    module.make_runtime = make_runtime
    monkeypatch.setitem(sys.modules, "zero_arg_egm_runtime_module", module)
    monkeypatch.setenv("CSD_EGM_HOME_DEVICE", "2")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "racer.csd",
            "--backend",
            "egm",
            "--egm-runtime-factory",
            "zero_arg_egm_runtime_module:make_runtime",
        ],
    )
    served = {}
    monkeypatch.setattr(
        csd_mod,
        "_serve",
        lambda address, authkey, backend, options, metadata_dir: served.update(
            backend=backend,
            options=options,
        ),
    )

    csd_mod.main()

    assert calls == ["called"]
    assert served["backend"] == "egm"
    assert served["options"]["runtime"] is runtime
    assert served["options"]["home_device"] == 2


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


def test_csd_atomic_metadata_commit_is_single_step_and_idempotent(tmp_path):
    class MetadataOnlyBackend:
        name = "fake"

        def capabilities(self):
            return {"backend": self.name, "daemon_owned": True}

        def list_chunks(self, tag):
            return []

        def free_tag(self, tag):
            pass

    daemon = csd_mod.CheckpointStorageDaemon(MetadataOnlyBackend(), metadata_dir=tmp_path)
    manifest = {
        "racer_manifest_kind": "megatron_tensor_tree",
        "checkpoint_tag": "checkpoint",
        "generation": "gen-1",
    }
    try:
        assert daemon.commit_metadata("meta", manifest)["created"] is True
        assert daemon.commit_metadata("meta", manifest)["created"] is False
        loaded = daemon.get_manifest("meta")
        assert loaded["generation"] == "gen-1"
        assert loaded["committed"] is True
        assert loaded["data_resident"] is True
        assert daemon.capabilities()["supports_atomic_metadata_commit"] is True
        assert list(tmp_path.glob("*.pt")) == []

        with pytest.raises(RuntimeError, match="different content"):
            daemon.commit_metadata("meta", {**manifest, "generation": "gen-2"})
        assert daemon.get_manifest("meta")["generation"] == "gen-1"
    finally:
        daemon.close()


def test_csd_restart_ignores_stale_metadata_only_pt_cache(tmp_path):
    class MetadataOnlyBackend:
        name = "fake"

        def capabilities(self):
            return {"backend": self.name, "daemon_owned": True}

        def list_chunks(self, tag):
            return []

    manifest = {"racer_manifest_kind": "marker", "generation": "gen-1"}
    first = csd_mod.CheckpointStorageDaemon(MetadataOnlyBackend(), metadata_dir=tmp_path)
    first.begin("meta", manifest, expected_chunks=0)
    stale_path = first._metadata_path("meta")
    assert stale_path.exists()
    assert first.manifest_store is not None
    first.manifest_store.commit_metadata_checkpoint("meta", manifest, backend="fake")
    first.close()

    reopened = csd_mod.CheckpointStorageDaemon(MetadataOnlyBackend(), metadata_dir=tmp_path)
    try:
        assert "meta" not in reopened._entries
        loaded = reopened.get_manifest("meta")
        assert loaded["generation"] == "gen-1"
        assert loaded["committed"] is True
    finally:
        reopened.close()


def test_csd_client_atomic_metadata_commit_uses_one_request(monkeypatch):
    client = CheckpointStorageDaemonClient("unused.sock", authkey="racer-csd")
    requests = []

    def fake_request(payload):
        requests.append(dict(payload))
        return {"tag": payload["tag"], "committed": True, "created": True}

    monkeypatch.setattr(client, "_request", fake_request)
    result = client.commit_metadata("meta", {"generation": "gen-1"})

    assert result["created"] is True
    assert requests == [
        {
            "op": "commit_metadata",
            "tag": "meta",
            "manifest": {"generation": "gen-1"},
        }
    ]


def test_csd_client_atomic_metadata_commit_falls_back_for_old_daemon(monkeypatch):
    client = CheckpointStorageDaemonClient("unused.sock", authkey="racer-csd")
    operations = []

    def fake_request(payload):
        operations.append(payload["op"])
        if payload["op"] == "commit_metadata":
            raise RuntimeError("unknown CSD op 'commit_metadata'")
        if payload["op"] == "begin":
            return {"tag": payload["tag"], "committed": False}
        return None

    monkeypatch.setattr(client, "_request", fake_request)
    result = client.commit_metadata("meta", {"generation": "gen-1"})

    assert result["created"] is True
    assert operations == ["commit_metadata", "begin", "put_manifest", "commit"]


def test_csd_without_metadata_dir_requires_every_expected_chunk_to_be_resident():
    class FakeBackend:
        name = "fake"

        def __init__(self):
            self.chunks = set()

        def capabilities(self):
            return {"backend": self.name, "daemon_owned": True}

        def list_chunks(self, tag):
            return sorted(self.chunks)

    backend = FakeBackend()
    daemon = csd_mod.CheckpointStorageDaemon(backend)
    daemon.begin(
        "dist",
        {"chunks": [{"chunk_id": "c0", "owner_rank": 0}]},
        expected_chunks=1,
    )
    daemon._entries["dist"]["chunks"]["c0"] = {"chunk_id": "c0", "nbytes": 8}

    with pytest.raises(RuntimeError, match="sealed_chunks=0, expected_chunks=1"):
        daemon.commit("dist")

    backend.chunks.add("c0")
    daemon.commit("dist")
    assert daemon.get_manifest("dist")["data_resident"] is True

    backend.chunks.clear()
    assert daemon.get_manifest("dist")["data_resident"] is False


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
    assert backend.poll(op_id) == "DONE"
    assert backend.list_chunks("tag") == []
    with pytest.raises(KeyError, match="not resident"):
        backend.metadata("tag", "c0")
    backend.wait(op_id)
    assert backend.metadata("tag", "c0")["nbytes"] == 123
    assert backend.list_chunks("tag") == ["c0"]
    assert backend.read_to_cuda_ipc("tag", "c0", {"nbytes": 123}) == "get-op"

    with pytest.raises(NotImplementedError, match="CPU/socket byte put"):
        backend.write_from_bytes("tag", "c1", b"abc")


def test_egm_backend_publishes_write_metadata_only_after_successful_completion():
    class EagerMetadataRuntime:
        def __init__(self):
            self.next_op = 0
            self.metadata_by_tag = {}
            self.fail_wait = set()
            self.poll_states = {}

        def write_from_cuda_ipc(self, tag, chunk_id, view, metadata):
            op_id = f"put-{self.next_op}"
            self.next_op += 1
            record = dict(metadata)
            record.update(
                {
                    "op_id": op_id,
                    "allocation_id": f"alloc-{op_id}",
                    "nbytes": int(view["nbytes"]),
                    "location": {
                        "allocation_id": f"alloc-{op_id}",
                        "offset": int(metadata.get("offset", 0)),
                    },
                }
            )
            # Deliberately expose the candidate early to verify that the
            # wrapper keeps its prior generation authoritative.
            self.metadata_by_tag.setdefault(tag, {})[chunk_id] = dict(record)
            self.poll_states[op_id] = "RUNNING"
            return op_id, record

        def read_to_cuda_ipc(self, tag, chunk_id, view):
            return "get-op"

        def wait(self, op_id):
            if op_id in self.fail_wait:
                raise RuntimeError(f"injected failure for {op_id}")

        def poll(self, op_id):
            return self.poll_states[op_id]

        def metadata(self, tag, chunk_id):
            return dict(self.metadata_by_tag[tag][chunk_id])

    runtime = EagerMetadataRuntime()
    backend = EgmBackend(runtime=runtime, pool_id="pool0")

    initial_op, _ = backend.write_from_cuda_ipc(
        "tag",
        "c0",
        {"nbytes": 8},
        {"owner_rank": 0, "offset": 0},
    )
    with pytest.raises(KeyError, match="not resident"):
        backend.metadata("tag", "c0")
    backend.wait(initial_op)
    initial = backend.metadata("tag", "c0")
    assert initial["owner_rank"] == 0
    assert initial["nbytes"] == 8
    assert initial["location"]["offset"] == 0

    failed_op, _ = backend.write_from_cuda_ipc(
        "tag",
        "c0",
        {"nbytes": 16},
        {"owner_rank": 4, "offset": 64},
    )
    assert backend.metadata("tag", "c0")["owner_rank"] == 0
    runtime.fail_wait.add(failed_op)
    with pytest.raises(RuntimeError, match="injected failure"):
        backend.wait(failed_op)
    after_failure = backend.metadata("tag", "c0")
    assert after_failure["owner_rank"] == 0
    assert after_failure["nbytes"] == 8
    assert after_failure["location"]["offset"] == 0

    polled_op, _ = backend.write_from_cuda_ipc(
        "tag",
        "c0",
        {"nbytes": 32},
        {"owner_rank": 5, "offset": 128},
    )
    assert backend.poll(polled_op) == "RUNNING"
    assert backend.metadata("tag", "c0")["owner_rank"] == 0
    runtime.poll_states[polled_op] = "DONE"
    assert backend.poll(polled_op) == "DONE"
    after_success = backend.metadata("tag", "c0")
    assert after_success["owner_rank"] == 5
    assert after_success["nbytes"] == 32
    assert after_success["location"]["offset"] == 128

    failed_poll_op, _ = backend.write_from_cuda_ipc(
        "tag",
        "c0",
        {"nbytes": 64},
        {"owner_rank": 6, "offset": 256},
    )
    runtime.poll_states[failed_poll_op] = "FAILED"
    assert backend.poll(failed_poll_op) == "FAILED"
    after_poll_failure = backend.metadata("tag", "c0")
    assert after_poll_failure["owner_rank"] == 5
    assert after_poll_failure["nbytes"] == 32
    assert after_poll_failure["location"]["offset"] == 128


def test_backend_future_resolution_does_not_publish_chunk_metadata_before_wait(monkeypatch):
    monkeypatch.setattr(csd_mod, "_visible_cuda_device_count", lambda: 0)
    new_metadata = {
        "generation": "new",
        "nbytes": 16,
        "checksum_type": "none",
        "location": {"offset": 64, "nbytes": 16},
    }

    class DelayedPublishBackend:
        name = "fake"

        def capabilities(self):
            return {"backend": self.name}

        def wait(self, op_id):
            assert op_id == "backend-op"

        def profile(self, op_id):
            return {}

        def metadata(self, tag, chunk_id):
            return dict(new_metadata)

        def update_metadata(self, tag, chunk_id, metadata):
            new_metadata.update(dict(metadata))

    daemon = csd_mod.CheckpointStorageDaemon(DelayedPublishBackend())
    future = csd_mod.Future()
    future.set_result(("backend-op", dict(new_metadata), {"daemon_put_backend_write_ms": 1.0}))
    daemon._entries["tag"] = {
        "tag": "tag",
        "chunks": {"c0": {"generation": "old", "nbytes": 8, "location": {"offset": 0}}},
    }
    daemon._async_ops["daemon-op"] = {
        "backend_op_id": None,
        "backend_future": future,
        "op_type": "PUT",
        "tag": "tag",
        "chunk_id": "c0",
        "metadata": {},
        "profile": {},
    }
    try:
        resolved = daemon._resolve_backend_future("daemon-op", dict(daemon._async_ops["daemon-op"]))
        assert resolved["metadata"]["generation"] == "new"
        assert daemon._entries["tag"]["chunks"]["c0"]["generation"] == "old"

        assert daemon.wait("daemon-op")["state"] == "DONE"
        assert daemon._entries["tag"]["chunks"]["c0"]["generation"] == "new"
    finally:
        daemon.close()


def test_pending_seal_batch_failure_requeues_tail_and_commit_recovers(monkeypatch, tmp_path):
    monkeypatch.setattr(csd_mod, "_visible_cuda_device_count", lambda: 0)
    chunk_ids = ["c0", "c1", "c2", "c3"]

    class ResidentBackend:
        name = "fake"

        def capabilities(self):
            return {"backend": self.name, "daemon_owned": True}

        def list_chunks(self, tag):
            return list(chunk_ids) if str(tag) == "tag" else []

    daemon = csd_mod.CheckpointStorageDaemon(ResidentBackend(), metadata_dir=tmp_path)
    manifest = {
        "tag": "tag",
        "chunks": [
            {"chunk_id": chunk_id, "owner_rank": index}
            for index, chunk_id in enumerate(chunk_ids)
        ],
    }
    daemon.begin("tag", manifest, expected_chunks=len(chunk_ids))
    assert daemon.manifest_store is not None

    seals = []
    for index, chunk_id in enumerate(chunk_ids):
        metadata = {"nbytes": 8, "valid_nbytes": 8, "owner_rank": index}
        daemon.manifest_store.reserve_chunk("tag", chunk_id, metadata, backend="fake")
        seals.append(
            {
                "chunk_id": chunk_id,
                "location": {"offset": index * 8, "nbytes": 8},
                "checksum_type": "none",
                "checksum": "",
                "nbytes": 8,
                "valid_nbytes": 8,
            }
        )
    with daemon._lock:
        daemon._entries["tag"]["chunks"] = {
            chunk_id: {"nbytes": 8, "valid_nbytes": 8}
            for chunk_id in chunk_ids
        }
        # c3 models a seal concurrently appended while the popped batch is
        # being flushed.
        daemon._pending_seals["tag"] = list(seals[:3])

    original_seal_chunk = daemon.manifest_store.seal_chunk
    calls = []
    injected = False

    def flaky_seal_chunk(tag, chunk_id, **kwargs):
        nonlocal injected
        calls.append(str(chunk_id))
        if str(chunk_id) == "c1" and not injected:
            injected = True
            with daemon._lock:
                daemon._pending_seals.setdefault("tag", []).append(seals[3])
            raise RuntimeError("injected seal failure")
        return original_seal_chunk(tag, chunk_id, **kwargs)

    monkeypatch.setattr(daemon.manifest_store, "seal_chunk", flaky_seal_chunk)
    try:
        with pytest.raises(RuntimeError, match="injected seal failure"):
            daemon.commit("tag")

        assert daemon._entries["tag"]["committed"] is False
        with pytest.raises(KeyError, match="not committed"):
            daemon.get_manifest("tag")
        with daemon._lock:
            assert [seal["chunk_id"] for seal in daemon._pending_seals["tag"]] == ["c1", "c2", "c3"]
        assert {
            record.chunk_id: record.state
            for record in daemon.manifest_store.list_chunks("tag")
        } == {"c0": "SEALED", "c1": "RESERVED", "c2": "RESERVED", "c3": "RESERVED"}

        daemon.commit("tag")

        assert calls == ["c0", "c1", "c1", "c2", "c3"]
        assert daemon._entries["tag"]["committed"] is True
        assert "tag" not in daemon._pending_seals
        loaded = daemon.get_manifest("tag")
        assert loaded["committed"] is True
        assert loaded["data_resident"] is True
    finally:
        daemon.close()


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


def _native_backend_with_pending_overwrite():
    backend = NativePinnedMemoryBackend.__new__(NativePinnedMemoryBackend)
    backend._lock = threading.RLock()
    backend._op_condition = threading.Condition(backend._lock)
    segment = csd_mod._NativeSegment(segment_id="seg0", ptr=0, nbytes=256, offset=96)
    backend._segment_by_id = {segment.segment_id: segment}
    backend._free_blocks = []
    old_chunk = csd_mod.BackendChunk(
        tensor=torch.empty(0, dtype=torch.uint8),
        metadata={"generation": "old"},
        nbytes=32,
        capacity_nbytes=32,
        host_ptr=0,
        segment_id=segment.segment_id,
        offset=0,
    )
    new_chunk = csd_mod.BackendChunk(
        tensor=torch.empty(0, dtype=torch.uint8),
        metadata={"generation": "new"},
        nbytes=32,
        capacity_nbytes=32,
        host_ptr=64,
        segment_id=segment.segment_id,
        offset=64,
    )
    op = csd_mod._NativeCopyOp(
        op_id="overwrite-op",
        device=0,
        stream=0,
        wait_start_event=1,
        copy_start_event=2,
        complete_event=3,
        remote_ptr=None,
        remote_event=None,
        stream_owned=False,
        publish_tag="tag",
        publish_chunk_id="chunk",
        pending_chunk=new_chunk,
    )
    backend._chunks = {"tag": {"chunk": old_chunk}}
    backend._ops = {op.op_id: op}
    return backend, op, old_chunk, new_chunk


def test_native_async_overwrite_publishes_after_copy_and_reclaims_old_location(monkeypatch):
    backend, op, old_chunk, new_chunk = _native_backend_with_pending_overwrite()

    class FakeCudaRuntime:
        def cudaEventSynchronize(self, _event):
            assert backend._chunks["tag"]["chunk"] is old_chunk
            return 0

        def cudaEventDestroy(self, _event):
            return 0

    monkeypatch.delenv("RACER_CSD_CUDA_EVENT_TIMING", raising=False)
    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda _device: None)

    backend.wait(op.op_id)

    assert backend._chunks["tag"]["chunk"] is new_chunk
    assert op.pending_chunk is None
    assert op.done is True
    assert op.error is None
    assert [(block.segment_id, block.offset, block.nbytes) for block in backend._free_blocks] == [
        ("seg0", 0, 32)
    ]


def test_native_async_overwrite_failure_rolls_back_new_location(monkeypatch):
    backend, op, old_chunk, _new_chunk = _native_backend_with_pending_overwrite()

    class FakeCudaRuntime:
        def cudaEventSynchronize(self, _event):
            assert backend._chunks["tag"]["chunk"] is old_chunk
            return 719

        def cudaEventDestroy(self, _event):
            return 0

    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda _device: None)

    with pytest.raises(RuntimeError, match="native pinned copy wait failed"):
        backend.wait(op.op_id)

    assert backend._chunks["tag"]["chunk"] is old_chunk
    assert op.pending_chunk is None
    assert op.done is True
    assert op.error is not None
    assert op.resources_released is True
    assert [(block.segment_id, block.offset, block.nbytes) for block in backend._free_blocks] == [
        ("seg0", 64, 32)
    ]


def test_native_async_reads_hold_retired_location_until_every_op_finishes(monkeypatch):
    backend, _write_op, old_chunk, new_chunk = _native_backend_with_pending_overwrite()
    backend._ops = {}
    next_event = 10

    class FakeCudaRuntime:
        def cudaEventRecord(self, _event, _stream):
            return 0

        def cudaStreamWaitEvent(self, _stream, _event, _flags):
            return 0

        def cudaMemcpyAsync(self, _dst, _src, _nbytes, _kind, _stream):
            return 0

        def cudaEventSynchronize(self, event):
            return 719 if int(event.value) == 22 else 0

        def cudaEventDestroy(self, _event):
            return 0

        def cudaIpcCloseMemHandle(self, _ptr):
            return 0

    def fake_new_stream_and_events(_device):
        nonlocal next_event
        base = next_event
        next_event += 10
        return 7, base, base + 1, base + 2, False

    monkeypatch.delenv("RACER_CSD_CUDA_EVENT_TIMING", raising=False)
    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda _device: None)
    monkeypatch.setattr(
        backend,
        "_open_ipc_mem",
        lambda _view: (1024, 99, "", 0.0, 0.0, True, 0.0),
    )
    monkeypatch.setattr(backend, "_new_stream_and_events", fake_new_stream_and_events)
    view = {"device": 0, "nbytes": 32, "base_offset": 0}

    first_read = backend._read_to_cuda_ipc_unserialized("tag", "chunk", view)
    second_read = backend._read_to_cuda_ipc_unserialized("tag", "chunk", view)
    assert old_chunk.read_refcount == 2

    with backend._lock:
        backend._publish_chunk_locked("tag", "chunk", new_chunk)

    assert backend._chunks["tag"]["chunk"] is new_chunk
    assert old_chunk.retired is True
    assert old_chunk.location_released is False
    assert backend._free_blocks == []

    backend.wait(first_read)
    assert old_chunk.read_refcount == 1
    assert old_chunk.location_released is False
    assert backend._free_blocks == []

    with pytest.raises(RuntimeError, match="native pinned copy wait failed"):
        backend.wait(second_read)
    assert old_chunk.read_refcount == 0
    assert old_chunk.location_released is True
    assert [(block.segment_id, block.offset, block.nbytes) for block in backend._free_blocks] == [
        ("seg0", 0, 32)
    ]


def _native_backend_for_unregistered_copy_failure():
    backend = NativePinnedMemoryBackend.__new__(NativePinnedMemoryBackend)
    backend._lock = threading.RLock()
    backend._op_condition = threading.Condition(backend._lock)
    backend._chunks = {}
    backend._ops = {}
    backend._free_blocks = []
    segment = csd_mod._NativeSegment(segment_id="seg0", ptr=4096, nbytes=256, offset=32)
    backend._segment_by_id = {segment.segment_id: segment}
    allocation = csd_mod._NativeAllocation(
        segment=segment,
        offset=0,
        nbytes=32,
        source="pool_bump",
        allocate_ms=0.0,
    )
    backend._allocate_location = lambda *_args, **_kwargs: allocation
    backend._open_ipc_mem = lambda _view: (8192, 81, "", 0.0, 0.0, True, 0.0)
    backend._new_stream_and_events = lambda _device: (71, 72, 73, 74, True)
    return backend, segment


def test_native_write_enqueue_failure_releases_all_unregistered_resources(monkeypatch):
    backend, _segment = _native_backend_for_unregistered_copy_failure()
    calls = {"synchronized": [], "events": [], "mem": [], "streams": []}

    class FakeCudaRuntime:
        def cudaEventRecord(self, _event, _stream):
            return 0

        def cudaStreamWaitEvent(self, _stream, _event, _flags):
            return 0

        def cudaMemcpyAsync(self, _dst, _src, _nbytes, _kind, _stream):
            return 719

        def cudaStreamSynchronize(self, stream):
            calls["synchronized"].append(int(stream.value))
            return 0

        def cudaEventDestroy(self, event):
            calls["events"].append(int(event.value))
            return 0

        def cudaIpcCloseMemHandle(self, ptr):
            calls["mem"].append(int(ptr.value))
            return 0

        def cudaStreamDestroy(self, stream):
            calls["streams"].append(int(stream.value))
            return 0

    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda _device: None)

    with pytest.raises(RuntimeError, match="cudaMemcpyAsync D2H"):
        backend.write_from_cuda_ipc("tag", "c0", {"device": 0, "nbytes": 32}, {})

    assert backend._ops == {}
    assert backend._chunks == {}
    assert [(block.segment_id, block.offset, block.nbytes) for block in backend._free_blocks] == [
        ("seg0", 0, 32)
    ]
    assert calls == {
        "synchronized": [71],
        "events": [81, 72, 73, 74],
        "mem": [8192],
        "streams": [71],
    }


def test_native_write_event_setup_failure_releases_ipc_and_allocation(monkeypatch):
    backend, _segment = _native_backend_for_unregistered_copy_failure()
    destroyed_events = []
    closed_mem = []
    backend._new_stream_and_events = lambda _device: (_ for _ in ()).throw(RuntimeError("event setup failed"))

    class FakeCudaRuntime:
        def cudaEventDestroy(self, event):
            destroyed_events.append(int(event.value))
            return 0

        def cudaIpcCloseMemHandle(self, ptr):
            closed_mem.append(int(ptr.value))
            return 0

    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda _device: None)

    with pytest.raises(RuntimeError, match="event setup failed"):
        backend.write_from_cuda_ipc("tag", "c0", {"device": 0, "nbytes": 32}, {})

    assert destroyed_events == [81]
    assert closed_mem == [8192]
    assert [(block.segment_id, block.offset, block.nbytes) for block in backend._free_blocks] == [
        ("seg0", 0, 32)
    ]


def test_native_read_enqueue_failure_releases_resources_and_read_lease(monkeypatch):
    backend, segment = _native_backend_for_unregistered_copy_failure()
    chunk = csd_mod.BackendChunk(
        tensor=torch.empty(0, dtype=torch.uint8),
        metadata={"nbytes": 32},
        nbytes=32,
        capacity_nbytes=32,
        host_ptr=segment.ptr,
        segment_id=segment.segment_id,
        offset=0,
    )
    backend._chunks = {"tag": {"c0": chunk}}
    calls = {"synchronized": [], "events": [], "mem": [], "streams": []}

    class FakeCudaRuntime:
        def cudaEventRecord(self, _event, _stream):
            return 0

        def cudaStreamWaitEvent(self, _stream, _event, _flags):
            return 0

        def cudaMemcpyAsync(self, _dst, _src, _nbytes, _kind, _stream):
            return 719

        def cudaStreamSynchronize(self, stream):
            calls["synchronized"].append(int(stream.value))
            return 0

        def cudaEventDestroy(self, event):
            calls["events"].append(int(event.value))
            return 0

        def cudaIpcCloseMemHandle(self, ptr):
            calls["mem"].append(int(ptr.value))
            return 0

        def cudaStreamDestroy(self, stream):
            calls["streams"].append(int(stream.value))
            return 0

    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda _device: None)

    with pytest.raises(RuntimeError, match="cudaMemcpyAsync H2D"):
        backend._read_to_cuda_ipc_unserialized("tag", "c0", {"device": 0, "nbytes": 32})

    assert chunk.read_refcount == 0
    assert chunk.retired is False
    assert chunk.location_released is False
    assert backend._ops == {}
    assert calls == {
        "synchronized": [71],
        "events": [81, 72, 73, 74],
        "mem": [8192],
        "streams": [71],
    }


def test_native_event_creation_failure_destroys_partial_event_set(monkeypatch):
    backend = NativePinnedMemoryBackend.__new__(NativePinnedMemoryBackend)
    backend._get_copy_stream = lambda _device: 71
    created = []
    destroyed = []

    class FakeCudaRuntime:
        def cudaEventCreateWithFlags(self, event_ref, _flags):
            if created:
                return 719
            event_ref._obj.value = 72
            created.append(72)
            return 0

        def cudaEventDestroy(self, event):
            destroyed.append(int(event.value))
            return 0

    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())

    with pytest.raises(RuntimeError, match="cudaEventCreateWithFlags"):
        backend._new_stream_and_events(0)

    assert created == [72]
    assert destroyed == [72]


def test_native_invalid_ipc_event_handle_releases_open_mem_reference(monkeypatch):
    backend = NativePinnedMemoryBackend.__new__(NativePinnedMemoryBackend)
    backend._ipc_lock = threading.RLock()
    backend._ipc_mem_cache_max_entries = 8
    view = {
        "device": 0,
        "producer_pid": -1,
        "producer_token": "producer",
        "mem_handle": bytes(64),
        "event_handle": b"invalid",
    }
    cache_key = backend._ipc_cache_key(view)
    backend._ipc_mem_cache = {
        cache_key: {
            "ptr": 8192,
            "refcount": 0,
            "device": 0,
            "keep_idle": False,
            "producer_pid": -1,
            "producer_token": "producer",
            "last_used_ns": 0,
        }
    }
    closed_mem = []

    class FakeCudaRuntime:
        def cudaIpcCloseMemHandle(self, ptr):
            closed_mem.append(int(ptr.value))
            return 0

    monkeypatch.setattr(csd_mod, "_load_cudart", lambda: FakeCudaRuntime())
    monkeypatch.setattr(csd_mod, "_cuda_set_device", lambda _device: None)

    with pytest.raises(RuntimeError, match="event handle must be 64 bytes"):
        backend._open_ipc_mem(view)

    assert cache_key not in backend._ipc_mem_cache
    assert closed_mem == [8192]


def test_delete_drains_pending_put_before_final_backend_free(monkeypatch):
    monkeypatch.setattr(csd_mod, "_visible_cuda_device_count", lambda: 0)
    future_result_called = threading.Event()

    class NotifyingFuture(csd_mod.Future):
        def result(self, timeout=None):
            future_result_called.set()
            return super().result(timeout=timeout)

    class DelayedExecutor:
        def __init__(self, **_kwargs):
            self.pending = []

        def submit(self, fn):
            future = NotifyingFuture()
            self.pending.append((future, fn))
            return future

        def complete_next(self):
            future, fn = self.pending.pop(0)
            future.set_result(fn())

        def shutdown(self, wait=True, cancel_futures=False):
            del wait, cancel_futures

    class PublishOnWaitBackend:
        name = "fake"

        def __init__(self):
            self.pending = {}
            self.chunks = {}
            self.wait_published = threading.Event()

        def capabilities(self):
            return {"backend": self.name, "daemon_owned": True}

        def write_from_cuda_ipc(self, tag, chunk_id, view, metadata):
            op_id = "backend-put"
            record = dict(metadata)
            record.update({"nbytes": int(view["nbytes"]), "checksum_type": "none"})
            self.pending[op_id] = (str(tag), str(chunk_id), record)
            return op_id, record

        def wait(self, op_id):
            tag, chunk_id, record = self.pending.pop(str(op_id))
            self.chunks.setdefault(tag, {})[chunk_id] = dict(record)
            self.wait_published.set()

        def profile(self, _op_id):
            return {}

        def free(self, tag, chunk_id):
            chunks = self.chunks.get(str(tag), {})
            chunks.pop(str(chunk_id), None)
            if not chunks:
                self.chunks.pop(str(tag), None)

        def free_tag(self, tag):
            self.chunks.pop(str(tag), None)

        def list_chunks(self, tag):
            return sorted(self.chunks.get(str(tag), {}))

    monkeypatch.setattr(csd_mod, "ThreadPoolExecutor", DelayedExecutor)
    backend = PublishOnWaitBackend()
    daemon = csd_mod.CheckpointStorageDaemon(backend)
    put = daemon.put_cuda_ipc("tag", "c0", {"nbytes": 32}, {})
    cancellation_marked = threading.Event()

    class CancellationNotifyingDict(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key == "cancelled_by_delete" and value:
                cancellation_marked.set()

    with daemon._lock:
        daemon._async_ops[put["op_id"]] = CancellationNotifyingDict(daemon._async_ops[put["op_id"]])

    wait_result = {}

    def run_wait():
        wait_result.update(daemon.wait(put["op_id"]))

    wait_thread = threading.Thread(target=run_wait)
    wait_thread.start()
    assert future_result_called.wait(timeout=2.0)
    deleted = threading.Event()

    def run_delete():
        daemon.delete("tag")
        deleted.set()

    delete_thread = threading.Thread(target=run_delete)
    delete_thread.start()
    assert cancellation_marked.wait(timeout=2.0)
    assert not deleted.is_set()

    daemon._put_executor.complete_next()
    wait_thread.join(timeout=2.0)
    delete_thread.join(timeout=2.0)

    assert not wait_thread.is_alive()
    assert deleted.is_set()
    assert wait_result["state"] == "FAILED"
    assert "cancelled because tag" in wait_result["error"]
    assert backend.wait_published.is_set()
    assert "tag" not in backend.chunks
    assert "tag" not in daemon._entries
    assert put["op_id"] not in daemon._async_ops
    daemon.close()


def test_daemon_poll_waits_for_backend_dispatch_before_polling_backend(monkeypatch):
    monkeypatch.setattr(csd_mod, "_visible_cuda_device_count", lambda: 0)

    class PollingBackend:
        name = "fake"

        def __init__(self):
            self.polled = []

        def capabilities(self):
            return {"backend": self.name, "daemon_owned": True}

        def poll(self, op_id):
            self.polled.append(op_id)
            return "RUNNING"

    backend = PollingBackend()
    daemon = csd_mod.CheckpointStorageDaemon(backend)
    dispatch = csd_mod.Future()
    with daemon._lock:
        daemon._async_ops["daemon-put"] = {
            "backend_op_id": None,
            "backend_future": dispatch,
            "op_type": "PUT",
            "tag": "tag",
            "chunk_id": "c0",
            "metadata": {},
            "profile": {},
        }

    assert daemon.poll("daemon-put") == {"op_id": "daemon-put", "state": "RUNNING"}
    assert backend.polled == []

    with daemon._lock:
        completion_lock = daemon._async_ops["daemon-put"]["completion_lock"]
    completion_lock.acquire()
    try:
        assert daemon.poll("daemon-put") == {"op_id": "daemon-put", "state": "RUNNING"}
        assert backend.polled == []
    finally:
        completion_lock.release()

    dispatch.set_result(("backend-put", {"nbytes": 8}, {}))
    assert daemon.poll("daemon-put") == {"op_id": "daemon-put", "state": "RUNNING"}
    assert backend.polled == ["backend-put"]
    with daemon._lock:
        assert daemon._async_ops["daemon-put"]["backend_op_id"] == "backend-put"
        daemon._async_ops.pop("daemon-put")
    daemon.close()


def test_daemon_terminal_failed_poll_runs_wait_cleanup(monkeypatch):
    monkeypatch.setattr(csd_mod, "_visible_cuda_device_count", lambda: 0)

    class FailedBackend:
        name = "fake"

        def capabilities(self):
            return {"backend": self.name, "daemon_owned": True}

        def poll(self, op_id):
            assert op_id == "backend-put"
            return "FAILED"

        def wait(self, op_id):
            assert op_id == "backend-put"
            raise RuntimeError("injected backend failure")

    daemon = csd_mod.CheckpointStorageDaemon(FailedBackend())
    with daemon._lock:
        daemon._async_ops["daemon-put"] = {
            "backend_op_id": "backend-put",
            "op_type": "PUT",
            "tag": "tag",
            "chunk_id": "c0",
            "metadata": {},
            "profile": {},
        }

    result = daemon.poll("daemon-put")

    assert result["state"] == "FAILED"
    assert "injected backend failure" in result["error"]
    with daemon._lock:
        assert "daemon-put" not in daemon._async_ops
    daemon.close()


def test_daemon_poll_reports_completed_dispatch_failure_like_wait(monkeypatch):
    monkeypatch.setattr(csd_mod, "_visible_cuda_device_count", lambda: 0)

    class Backend:
        name = "fake"

        def capabilities(self):
            return {"backend": self.name, "daemon_owned": True}

    daemon = csd_mod.CheckpointStorageDaemon(Backend())
    dispatch = csd_mod.Future()
    dispatch.set_exception(RuntimeError("injected dispatch failure"))
    with daemon._lock:
        daemon._async_ops["daemon-put"] = {
            "backend_op_id": None,
            "backend_future": dispatch,
            "op_type": "PUT",
            "tag": "tag",
            "chunk_id": "c0",
            "metadata": {},
            "profile": {},
        }

    result = daemon.poll("daemon-put")

    assert result["state"] == "FAILED"
    assert "injected dispatch failure" in result["error"]
    with daemon._lock:
        assert "daemon-put" not in daemon._async_ops
    daemon.close()


def test_client_terminal_poll_releases_pending_cuda_view(monkeypatch):
    client = CheckpointStorageDaemonClient("unused.sock", authkey="racer-csd")

    class FakeView:
        def __init__(self):
            self.materialized = 0
            self.released = 0

        def materialize_after_read(self):
            self.materialized += 1

        def release(self):
            self.released += 1

    view = FakeView()
    client._pending_cuda_views["op0"] = view
    monkeypatch.setattr(client, "_request", lambda _payload: {"op_id": "op0", "state": "DONE"})

    assert client.poll("op0")["state"] == "DONE"
    assert view.materialized == 1
    assert view.released == 1
    assert "op0" not in client._pending_cuda_views


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
