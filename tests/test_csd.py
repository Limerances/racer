import ctypes
import os
import subprocess
import sys

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

    with pytest.raises(RuntimeError, match="put_chunk is disabled"):
        client.put_chunk("tag", "c0", payload)
    with pytest.raises(RuntimeError, match="put is disabled"):
        client.put("tag", "c0", payload)
    with pytest.raises(RuntimeError, match="get_chunk is disabled"):
        client.get_chunk("tag", "c0")
    with pytest.raises(RuntimeError, match="get is disabled"):
        client.get("tag", "c0")


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
