import hashlib
import multiprocessing as mp
import os
import traceback

import pytest
import torch

import racer
from racer.csd import CheckpointStorageDaemonClient


def _pattern(nbytes: int, seed: int) -> torch.Tensor:
    return (torch.arange(int(nbytes), dtype=torch.int64) + int(seed)).remainder(251).to(torch.uint8)


def _writer(address, authkey: bytes, tag: str, nbytes: int, queue) -> None:
    try:
        torch.cuda.set_device(0)
        client = CheckpointStorageDaemonClient(address, authkey=authkey)
        manifest = {
            "tag": tag,
            "k": 1,
            "m": 0,
            "train_ranks": [0],
            "spare_ranks": [],
            "chunks": [
                {"chunk_id": "c0", "row": 0, "owner_rank": 0, "num_bytes": int(nbytes)},
                {"chunk_id": "c1", "row": 1, "owner_rank": 0, "num_bytes": int(nbytes)},
            ],
        }
        client.begin(tag, manifest, expected_chunks=2)
        futures = []
        checksums = []
        for index in range(2):
            cpu = _pattern(nbytes, index)
            checksums.append(hashlib.sha256(cpu.numpy().tobytes()).hexdigest())
            gpu = cpu.to("cuda:0")
            op_id = client.put_cuda_tensor(
                tag,
                f"c{index}",
                gpu,
                {"row": index, "owner_rank": 0, "writer_rank": 0, "nbytes": int(nbytes)},
            )
            futures.append((op_id, gpu))
        for op_id, _gpu in futures:
            client.wait(op_id)
        client.put_manifest(tag, manifest)
        client.commit(tag)
        readbacks = []
        for index in range(2):
            dst = torch.empty(int(nbytes), dtype=torch.uint8, device="cuda:0")
            op_id = client.read_into_cuda_tensor(tag, f"c{index}", dst)
            readbacks.append((op_id, dst))
        host_checksums = []
        for op_id, dst in readbacks:
            client.wait(op_id)
            host_checksums.append(hashlib.sha256(dst.cpu().numpy().tobytes()).hexdigest())
        if host_checksums != checksums:
            raise AssertionError(f"daemon host checksum mismatch: observed={host_checksums}, expected={checksums}")
        queue.put(("ok", checksums))
    except BaseException:
        queue.put(("error", traceback.format_exc()))


def _reader(address, authkey: bytes, tag: str, nbytes: int, expected_checksums, queue) -> None:
    try:
        torch.cuda.set_device(0)
        client = CheckpointStorageDaemonClient(address, authkey=authkey)
        manifest = client.get_manifest(tag)
        if not manifest.get("committed"):
            raise AssertionError("manifest is not committed")
        futures = []
        for index in range(2):
            dst = torch.empty(int(nbytes), dtype=torch.uint8, device="cuda:0")
            op_id = client.read_into_cuda_tensor(tag, f"c{index}", dst)
            futures.append((op_id, dst))
        observed = []
        for op_id, dst in futures:
            client.wait(op_id)
            observed.append(hashlib.sha256(dst.cpu().numpy().tobytes()).hexdigest())
        if observed != list(expected_checksums):
            raise AssertionError(f"checksum mismatch: observed={observed}, expected={expected_checksums}")
        queue.put(("ok", observed))
    except BaseException:
        queue.put(("error", traceback.format_exc()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_native_pinned_csd_survives_training_process_restart(tmp_path):
    socket_path = tmp_path / "csd-native.sock"
    try:
        daemon = racer.start_checkpoint_storage_daemon(
            socket_path=socket_path,
            metadata_dir=tmp_path / "metadata",
            backend="native_pinned",
            backend_options={"segment_bytes": 8 * 1024 * 1024, "device": 0},
            ready_timeout_s=30.0,
        )
    except RuntimeError as exc:
        pytest.skip(f"native_pinned CSD cannot start in this environment: {exc}")

    try:
        ctx = mp.get_context("spawn")
        tag = "native-restart"
        nbytes = 1 * 1024 * 1024

        writer_queue = ctx.Queue()
        writer = ctx.Process(target=_writer, args=(str(socket_path), daemon.authkey, tag, nbytes, writer_queue))
        writer.start()
        writer.join(timeout=60.0)
        assert writer.exitcode == 0
        writer_status, writer_payload = writer_queue.get(timeout=5.0)
        assert writer_status == "ok", writer_payload
        assert daemon.process.is_alive()

        reader_queue = ctx.Queue()
        reader = ctx.Process(
            target=_reader,
            args=(str(socket_path), daemon.authkey, tag, nbytes, writer_payload, reader_queue),
        )
        reader.start()
        reader.join(timeout=60.0)
        assert reader.exitcode == 0
        reader_status, reader_payload = reader_queue.get(timeout=5.0)
        assert reader_status == "ok", reader_payload

        daemon.client.delete(tag)
    finally:
        daemon.shutdown()
