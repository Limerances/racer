#!/usr/bin/env python3
"""Layered CSD/native-pinned storage benchmark.

Outputs JSON lines by default: one record per chunk plus one final summary.
The modes intentionally separate CUDA copy, CUDA IPC, staging, checksum,
SQLite, and socket/RPC overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any

import torch

import racer
from racer.csd import (
    CheckpointStorageDaemonClient,
    _CUDA_EVENT_DISABLE_TIMING,
    _CUDA_HOST_ALLOC_DEFAULT,
    _CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
    _CUDA_MEMCPY_DEFAULT,
    _CUDA_MEMCPY_DEVICE_TO_DEVICE,
    _CUDA_STREAM_NON_BLOCKING,
    _CudaIpcEventHandle,
    _CudaIpcMemHandle,
    _cuda_check,
    _cuda_event_elapsed_ms,
    _cuda_set_device,
    _ipc_handle_bytes_from_ptr,
    _load_cudart,
    _record_interprocess_event,
    export_cuda_ipc_view,
)


PHASES = [
    "export_ipc_us",
    "staging_alloc_us",
    "staging_copy_ms",
    "socket_rpc_us",
    "daemon_allocate_ms",
    "daemon_ipc_open_us",
    "daemon_event_wait_ms",
    "daemon_memcpy_ms_cuda_event",
    "daemon_memcpy_ms_wall",
    "checksum_ms",
    "sqlite_ms",
    "ack_us",
    "total_ms",
]
SUMMARY_ONLY = False


def _now_us() -> float:
    return time.perf_counter() * 1_000_000.0


def _require_cuda(device: int) -> Any:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cudart = _load_cudart()
    if cudart is None:
        raise SystemExit("CUDA runtime is required")
    _cuda_set_device(int(device))
    return cudart


def _make_stream(cudart: Any) -> int:
    ptr = __import__("ctypes").c_void_p()
    import ctypes

    _cuda_check(
        cudart.cudaStreamCreateWithFlags(ctypes.byref(ptr), ctypes.c_uint(_CUDA_STREAM_NON_BLOCKING)),
        "cudaStreamCreateWithFlags failed",
    )
    return int(ptr.value)


def _make_event(cudart: Any, flags: int = 0) -> int:
    import ctypes

    event = ctypes.c_void_p()
    _cuda_check(cudart.cudaEventCreateWithFlags(ctypes.byref(event), ctypes.c_uint(flags)), "cudaEventCreateWithFlags failed")
    return int(event.value)


def _malloc_device(cudart: Any, nbytes: int) -> int:
    import ctypes

    ptr = ctypes.c_void_p()
    _cuda_check(cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes))), "cudaMalloc failed")
    return int(ptr.value)


def _malloc_host(cudart: Any, nbytes: int) -> int:
    import ctypes

    ptr = ctypes.c_void_p()
    _cuda_check(
        cudart.cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes)), ctypes.c_uint(_CUDA_HOST_ALLOC_DEFAULT)),
        "cudaHostAlloc failed",
    )
    return int(ptr.value)


def _timed_copy(cudart: Any, dst: int, src: int, nbytes: int, stream: int, kind: int = _CUDA_MEMCPY_DEFAULT) -> tuple[float, float]:
    import ctypes

    start_event = _make_event(cudart)
    end_event = _make_event(cudart)
    wall_start = time.perf_counter()
    _cuda_check(cudart.cudaEventRecord(ctypes.c_void_p(start_event), ctypes.c_void_p(stream)), "cudaEventRecord start failed")
    _cuda_check(
        cudart.cudaMemcpyAsync(
            ctypes.c_void_p(int(dst)),
            ctypes.c_void_p(int(src)),
            ctypes.c_size_t(int(nbytes)),
            ctypes.c_int(int(kind)),
            ctypes.c_void_p(int(stream)),
        ),
        "cudaMemcpyAsync failed",
    )
    _cuda_check(cudart.cudaEventRecord(ctypes.c_void_p(end_event), ctypes.c_void_p(stream)), "cudaEventRecord end failed")
    _cuda_check(cudart.cudaEventSynchronize(ctypes.c_void_p(end_event)), "cudaEventSynchronize benchmark copy failed")
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    event_ms = _cuda_event_elapsed_ms(start_event, end_event)
    cudart.cudaEventDestroy(ctypes.c_void_p(start_event))
    cudart.cudaEventDestroy(ctypes.c_void_p(end_event))
    return event_ms, wall_ms


def _open_mem_handle(cudart: Any, handle_bytes: bytes) -> int:
    import ctypes

    handle = _CudaIpcMemHandle()
    ctypes.memmove(ctypes.byref(handle), bytes(handle_bytes), 64)
    ptr = ctypes.c_void_p()
    _cuda_check(
        cudart.cudaIpcOpenMemHandle(ctypes.byref(ptr), handle, ctypes.c_uint(_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)),
        "cudaIpcOpenMemHandle failed",
    )
    return int(ptr.value)


def _open_event_handle(cudart: Any, handle_bytes: bytes) -> int:
    import ctypes

    handle = _CudaIpcEventHandle()
    ctypes.memmove(ctypes.byref(handle), bytes(handle_bytes), 64)
    event = ctypes.c_void_p()
    _cuda_check(cudart.cudaIpcOpenEventHandle(ctypes.byref(event), handle), "cudaIpcOpenEventHandle failed")
    return int(event.value)


def _destroy_quiet(cudart: Any, name: str, value: int | None) -> None:
    if not value:
        return
    import ctypes

    try:
        getattr(cudart, name)(ctypes.c_void_p(int(value)))
    except Exception:
        pass


def _base_row(mode: str, chunk_index: int, chunk_nbytes: int) -> dict[str, Any]:
    row = {"record_type": "chunk", "mode": mode, "chunk_index": int(chunk_index), "chunk_mib": chunk_nbytes / 1024 / 1024}
    for phase in PHASES:
        row[phase] = 0.0
    return row


def _emit(record: dict[str, Any]) -> None:
    if SUMMARY_ONLY and record.get("record_type") == "chunk":
        return
    print(json.dumps(record, sort_keys=True), flush=True)


def _summary(mode: str, rows: list[dict[str, Any]], total_wall_ms: float, total_bytes: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "record_type": "summary",
        "mode": mode,
        "chunks": len(rows),
        "total_gib": total_bytes / (1024**3),
        "total_wall_ms": float(total_wall_ms),
        "effective_gib_s": (total_bytes / (1024**3)) / max(total_wall_ms / 1000.0, 1e-9),
    }
    copy_ms = sum(float(row.get("daemon_memcpy_ms_cuda_event", 0.0)) for row in rows)
    summary["copy_only_gib_s"] = (total_bytes / (1024**3)) / max(copy_ms / 1000.0, 1e-9)
    for phase in PHASES:
        values = [float(row.get(phase, 0.0)) for row in rows]
        if not values:
            continue
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        summary[f"{phase}_p50"] = statistics.median(values)
        summary[f"{phase}_p95"] = ordered[p95_index]
        summary[f"{phase}_max"] = max(values)
    return summary


def mode_local_pinned_d2h_only(args: argparse.Namespace, mode: str = "local_pinned_d2h_only") -> list[dict[str, Any]]:
    cudart = _require_cuda(args.device)
    import ctypes

    chunk_nbytes = int(args.chunk_mib) * 1024 * 1024
    total_nbytes = int(args.num_chunks) * chunk_nbytes
    src = _malloc_device(cudart, total_nbytes)
    dst = _malloc_host(cudart, total_nbytes)
    stream = _make_stream(cudart)
    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    try:
        for index in range(int(args.num_chunks)):
            row = _base_row(mode, index, chunk_nbytes)
            event_ms, wall_ms = _timed_copy(
                cudart,
                dst + index * chunk_nbytes,
                src + index * chunk_nbytes,
                chunk_nbytes,
                stream,
            )
            row["daemon_memcpy_ms_cuda_event"] = event_ms
            row["daemon_memcpy_ms_wall"] = wall_ms
            row["total_ms"] = wall_ms
            _emit(row)
            rows.append(row)
    finally:
        cudart.cudaStreamDestroy(ctypes.c_void_p(stream))
        cudart.cudaFree(ctypes.c_void_p(src))
        cudart.cudaFreeHost(ctypes.c_void_p(dst))
    _emit(_summary(mode, rows, (time.perf_counter() - wall_start) * 1000.0, total_nbytes))
    return rows


def _daemon_no_ipc_worker(conn, cfg: dict[str, Any]) -> None:
    global SUMMARY_ONLY
    SUMMARY_ONLY = bool(cfg.get("summary_only", False))

    class Obj:
        pass

    args = Obj()
    args.device = cfg["device"]
    args.chunk_mib = cfg["chunk_mib"]
    args.num_chunks = cfg["num_chunks"]
    rows = mode_local_pinned_d2h_only(args, mode="daemon_pinned_d2h_no_ipc")
    conn.send(rows)
    conn.close()


def mode_daemon_pinned_d2h_no_ipc(args: argparse.Namespace) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    cfg = dict(vars(args))
    cfg["summary_only"] = bool(args.summary_only)
    proc = ctx.Process(target=_daemon_no_ipc_worker, args=(child, cfg))
    proc.start()
    rows = parent.recv()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"daemon worker failed with exitcode={proc.exitcode}")
    return rows


def _ipc_copy_worker(conn, cfg: dict[str, Any]) -> None:
    global SUMMARY_ONLY
    SUMMARY_ONLY = bool(cfg.get("summary_only", False))

    import ctypes

    mode = str(cfg["mode"])
    cudart = _require_cuda(int(cfg["device"]))
    chunk_nbytes = int(cfg["chunk_nbytes"])
    num_chunks = int(cfg["num_chunks"])
    src_handle = bytes(cfg["src_handle"])
    event_handle = bytes(cfg["event_handle"])
    per_chunk_open = bool(cfg.get("per_chunk_open", False))
    checksum = bool(cfg.get("checksum", False))
    views = cfg.get("views")
    dst = _malloc_host(cudart, chunk_nbytes * num_chunks)
    stream = _make_stream(cudart)
    rows: list[dict[str, Any]] = []
    remote_ptr: int | None = None
    shared_remote_event = _open_event_handle(cudart, event_handle)
    try:
        if not per_chunk_open:
            start = _now_us()
            first_handle = src_handle if views is None else bytes(views[0]["mem_handle"])
            remote_ptr = _open_mem_handle(cudart, first_handle)
            cached_open_us = _now_us() - start
        else:
            cached_open_us = 0.0
        for index in range(num_chunks):
            view = None if views is None else views[index]
            handle = src_handle if view is None else bytes(view["mem_handle"])
            offset = index * chunk_nbytes if view is None else int(view["offset"])
            row = _base_row(mode, index, chunk_nbytes)
            if view is not None:
                row["staging_alloc_us"] = float(view.get("staging_alloc_us", 0.0))
                row["staging_copy_ms"] = float(view.get("staging_copy_ms", 0.0))
            if per_chunk_open:
                open_start = _now_us()
                ptr = _open_mem_handle(cudart, handle)
                row["daemon_ipc_open_us"] = _now_us() - open_start
            else:
                ptr = int(remote_ptr)
                row["daemon_ipc_open_us"] = cached_open_us if index == 0 else 0.0
            if view is not None:
                remote_event = _open_event_handle(cudart, bytes(view["event_handle"]))
            else:
                remote_event = shared_remote_event
            wait_start = _make_event(cudart)
            copy_start = _make_event(cudart)
            done = _make_event(cudart)
            wall_start = time.perf_counter()
            _cuda_check(cudart.cudaEventRecord(ctypes.c_void_p(wait_start), ctypes.c_void_p(stream)), "record wait_start")
            _cuda_check(cudart.cudaStreamWaitEvent(ctypes.c_void_p(stream), ctypes.c_void_p(remote_event), ctypes.c_uint(0)), "stream wait ipc event")
            _cuda_check(cudart.cudaEventRecord(ctypes.c_void_p(copy_start), ctypes.c_void_p(stream)), "record copy_start")
            _cuda_check(
                cudart.cudaMemcpyAsync(
                    ctypes.c_void_p(dst + index * chunk_nbytes),
                    ctypes.c_void_p(ptr + offset),
                    ctypes.c_size_t(chunk_nbytes),
                    ctypes.c_int(_CUDA_MEMCPY_DEFAULT),
                    ctypes.c_void_p(stream),
                ),
                "ipc D2H copy",
            )
            _cuda_check(cudart.cudaEventRecord(ctypes.c_void_p(done), ctypes.c_void_p(stream)), "record done")
            _cuda_check(cudart.cudaEventSynchronize(ctypes.c_void_p(done)), "sync done")
            row["daemon_memcpy_ms_wall"] = (time.perf_counter() - wall_start) * 1000.0
            row["daemon_event_wait_ms"] = _cuda_event_elapsed_ms(wait_start, copy_start)
            row["daemon_memcpy_ms_cuda_event"] = _cuda_event_elapsed_ms(copy_start, done)
            if checksum:
                checksum_start = time.perf_counter()
                hashlib.sha256(ctypes.string_at(ctypes.c_void_p(dst + index * chunk_nbytes), chunk_nbytes)).hexdigest()
                row["checksum_ms"] = (time.perf_counter() - checksum_start) * 1000.0
            row["total_ms"] = row["daemon_memcpy_ms_wall"] + row["checksum_ms"] + row["daemon_ipc_open_us"] / 1000.0
            for event in (wait_start, copy_start, done):
                cudart.cudaEventDestroy(ctypes.c_void_p(event))
            if view is not None:
                cudart.cudaEventDestroy(ctypes.c_void_p(remote_event))
            if per_chunk_open:
                cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))
            _emit(row)
            rows.append(row)
    finally:
        if remote_ptr is not None:
            cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(remote_ptr))
        _destroy_quiet(cudart, "cudaEventDestroy", shared_remote_event)
        cudart.cudaStreamDestroy(ctypes.c_void_p(stream))
        cudart.cudaFreeHost(ctypes.c_void_p(dst))
    conn.send(rows)
    conn.close()


def _run_ipc_worker(args: argparse.Namespace, mode: str, *, per_chunk_open: bool, checksum: bool = False, views=None, src_handle=None, event_handle=None) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    cfg = {
        "mode": mode,
        "device": int(args.device),
        "chunk_nbytes": int(args.chunk_mib) * 1024 * 1024,
        "num_chunks": int(args.num_chunks),
        "per_chunk_open": per_chunk_open,
        "checksum": checksum,
        "views": views,
        "src_handle": src_handle,
        "event_handle": event_handle,
        "summary_only": bool(args.summary_only),
    }
    proc = ctx.Process(target=_ipc_copy_worker, args=(child, cfg))
    proc.start()
    rows = parent.recv()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"ipc worker failed with exitcode={proc.exitcode}")
    return rows


def _make_source_slab(args: argparse.Namespace) -> tuple[int, bytes, bytes, int]:
    cudart = _require_cuda(args.device)
    total_nbytes = int(args.num_chunks) * int(args.chunk_mib) * 1024 * 1024
    src = _malloc_device(cudart, total_nbytes)
    stream = _make_stream(cudart)
    event_handle, event = _record_interprocess_event(int(args.device), stream)
    return src, _ipc_handle_bytes_from_ptr(src), event_handle, stream


def mode_ipc_cached_or_open(args: argparse.Namespace, mode: str, *, per_chunk_open: bool, checksum: bool = False) -> list[dict[str, Any]]:
    cudart = _require_cuda(args.device)
    import ctypes

    src, handle, event_handle, stream = _make_source_slab(args)
    try:
        rows = _run_ipc_worker(args, mode, per_chunk_open=per_chunk_open, checksum=checksum, src_handle=handle, event_handle=event_handle)
        _emit(_summary(mode, rows, sum(row["total_ms"] for row in rows), int(args.num_chunks) * int(args.chunk_mib) * 1024 * 1024))
        return rows
    finally:
        cudart.cudaStreamDestroy(ctypes.c_void_p(stream))
        cudart.cudaFree(ctypes.c_void_p(src))


def mode_staging(args: argparse.Namespace, mode: str, *, reuse_slab: bool) -> list[dict[str, Any]]:
    cudart = _require_cuda(args.device)
    import ctypes

    chunk_nbytes = int(args.chunk_mib) * 1024 * 1024
    total_nbytes = int(args.num_chunks) * chunk_nbytes
    src = _malloc_device(cudart, total_nbytes)
    src_stream = _make_stream(cudart)
    views: list[dict[str, Any]] = []
    staging_ptrs: list[int] = []
    staging_slab = None
    try:
        if reuse_slab:
            alloc_start = _now_us()
            staging_slab = _malloc_device(cudart, total_nbytes)
            slab_alloc_us = _now_us() - alloc_start
            slab_handle = _ipc_handle_bytes_from_ptr(staging_slab)
        for index in range(int(args.num_chunks)):
            if reuse_slab:
                staging_ptr = int(staging_slab) + index * chunk_nbytes
                handle = slab_handle
                offset = index * chunk_nbytes
                alloc_us = slab_alloc_us if index == 0 else 0.0
            else:
                alloc_start = _now_us()
                staging_ptr = _malloc_device(cudart, chunk_nbytes)
                alloc_us = _now_us() - alloc_start
                handle = _ipc_handle_bytes_from_ptr(staging_ptr)
                offset = 0
                staging_ptrs.append(staging_ptr)
            start_event = _make_event(cudart)
            end_event = _make_event(cudart)
            _cuda_check(cudart.cudaEventRecord(ctypes.c_void_p(start_event), ctypes.c_void_p(src_stream)), "record staging start")
            _cuda_check(
                cudart.cudaMemcpyAsync(
                    ctypes.c_void_p(staging_ptr),
                    ctypes.c_void_p(src + index * chunk_nbytes),
                    ctypes.c_size_t(chunk_nbytes),
                    ctypes.c_int(_CUDA_MEMCPY_DEVICE_TO_DEVICE),
                    ctypes.c_void_p(src_stream),
                ),
                "staging D2D copy",
            )
            _cuda_check(cudart.cudaEventRecord(ctypes.c_void_p(end_event), ctypes.c_void_p(src_stream)), "record staging end")
            _cuda_check(cudart.cudaEventSynchronize(ctypes.c_void_p(end_event)), "sync staging copy")
            event_handle, ready_event = _record_interprocess_event(int(args.device), src_stream)
            views.append(
                {
                    "mem_handle": handle,
                    "event_handle": event_handle,
                    "offset": offset,
                    "staging_alloc_us": alloc_us,
                    "staging_copy_ms": _cuda_event_elapsed_ms(start_event, end_event),
                }
            )
            for event in (start_event, end_event, ready_event):
                cudart.cudaEventDestroy(ctypes.c_void_p(event))
        rows = _run_ipc_worker(
            args,
            mode,
            per_chunk_open=not reuse_slab,
            checksum=False,
            views=views,
            src_handle=views[0]["mem_handle"],
            event_handle=views[0]["event_handle"],
        )
        _emit(_summary(mode, rows, sum(row["total_ms"] for row in rows), total_nbytes))
        return rows
    finally:
        cudart.cudaStreamDestroy(ctypes.c_void_p(src_stream))
        cudart.cudaFree(ctypes.c_void_p(src))
        if staging_slab is not None:
            cudart.cudaFree(ctypes.c_void_p(staging_slab))
        for ptr in staging_ptrs:
            cudart.cudaFree(ctypes.c_void_p(ptr))


def mode_full_csd_put(args: argparse.Namespace) -> list[dict[str, Any]]:
    chunk_nbytes = int(args.chunk_mib) * 1024 * 1024
    total_nbytes = int(args.num_chunks) * chunk_nbytes
    csd_total_bytes = total_nbytes if args.csd_total_bytes is None else int(args.csd_total_bytes)
    torch.cuda.set_device(int(args.device))
    chunks = [torch.empty(chunk_nbytes, dtype=torch.uint8, device=f"cuda:{int(args.device)}") for _ in range(int(args.num_chunks))]
    torch.cuda.synchronize(int(args.device))
    with tempfile.TemporaryDirectory(prefix="racer-csd-layers-") as tmp:
        tmp_path = Path(tmp)
        daemon = racer.start_checkpoint_storage_daemon(
            socket_path=tmp_path / "csd.sock",
            metadata_dir=(tmp_path / "metadata") if args.include_sqlite else None,
            backend="native_pinned",
            backend_options={
                "total_bytes": csd_total_bytes,
                "segment_bytes": max(chunk_nbytes, 256 * 1024 * 1024),
                "device": int(args.device),
            },
        )
        client = CheckpointStorageDaemonClient(str(tmp_path / "csd.sock"), authkey=daemon.authkey)
        tag = f"full-{time.time_ns()}"
        manifest = {
            "tag": tag,
            "k": 1,
            "m": 0,
            "train_ranks": [0],
            "spare_ranks": [],
            "chunks": [{"chunk_id": f"c{i}", "row": i, "owner_rank": 0, "num_bytes": chunk_nbytes} for i in range(int(args.num_chunks))],
        }
        client.begin(tag, manifest, expected_chunks=int(args.num_chunks))
        rows: list[dict[str, Any]] = []
        pending: list[tuple[str, Any, dict[str, Any], float]] = []
        wall_start = time.perf_counter()
        wall_ms = 0.0
        try:
            for index, tensor in enumerate(chunks):
                row = _base_row("full_csd_put", index, chunk_nbytes)
                row["csd_total_bytes"] = csd_total_bytes
                total_start = time.perf_counter()
                export_start = _now_us()
                view = export_cuda_ipc_view(tensor)
                row["export_ipc_us"] = _now_us() - export_start
                export_profile = dict(view.profile or {})
                row["staging_alloc_us"] = float(export_profile.get("staging_alloc_us", 0.0))
                row["direct_ipc_error"] = export_profile.get("direct_ipc_error", "")
                row["staging_kind"] = export_profile.get("staging_kind", "")
                metadata = {
                    "row": index,
                    "owner_rank": 0,
                    "writer_rank": 0,
                    "nbytes": chunk_nbytes,
                    "checksum_type": args.checksum_type,
                    "manifest_update_mode": args.manifest_update_mode,
                }
                rpc_start = _now_us()
                result = client._request(
                    {
                        "op": "put_cuda_ipc",
                        "tag": tag,
                        "chunk_id": f"c{index}",
                        "metadata": metadata,
                        "view": view.as_request(),
                    }
                )
                row["socket_rpc_us"] = _now_us() - rpc_start
                pending.append((str(result["op_id"]), view, row, total_start))
                if len(pending) >= int(args.pipeline_depth):
                    _wait_full(client, pending.pop(0), rows)
            while pending:
                _wait_full(client, pending.pop(0), rows)
            client.put_manifest(tag, manifest)
            client.commit(tag)
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
        finally:
            daemon.shutdown()
        _emit(_summary("full_csd_put", rows, wall_ms, total_nbytes))
        return rows


def _wait_full(client: CheckpointStorageDaemonClient, item, rows: list[dict[str, Any]]) -> None:
    op_id, view, row, total_start = item
    try:
        ack_start = _now_us()
        result = client._request({"op": "wait", "op_id": op_id})
        row["ack_us"] = _now_us() - ack_start
        profile = dict(result.get("profile") or {})
        export_profile = dict(profile.get("export_profile") or {})
        row["staging_alloc_us"] = float(export_profile.get("staging_alloc_us", row.get("staging_alloc_us", 0.0)))
        row["daemon_allocate_ms"] = float(profile.get("daemon_allocate_ms", 0.0))
        row["daemon_allocate_source"] = str(profile.get("daemon_allocate_source", ""))
        row["daemon_ipc_open_us"] = float(profile.get("daemon_ipc_open_us", 0.0))
        row["daemon_event_wait_ms"] = float(profile.get("daemon_event_wait_ms", 0.0))
        row["daemon_memcpy_ms_cuda_event"] = float(profile.get("daemon_memcpy_ms_cuda_event", 0.0))
        row["daemon_memcpy_ms_wall"] = float(profile.get("daemon_memcpy_ms_wall", 0.0))
        row["checksum_ms"] = float(profile.get("checksum_ms", 0.0))
        row["sqlite_ms"] = float(profile.get("sqlite_ms", 0.0))
        row["ipc_opened_new"] = bool(profile.get("daemon_ipc_opened_new", False))
        row["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        _emit(row)
        rows.append(row)
    finally:
        view.release()


def run(args: argparse.Namespace) -> None:
    modes = [args.mode] if args.mode != "all" else [
        "local_pinned_d2h_only",
        "daemon_pinned_d2h_no_ipc",
        "daemon_pinned_d2h_with_ipc_cached",
        "daemon_pinned_d2h_with_ipc_open_per_chunk",
        "daemon_pinned_d2h_with_staging_e1",
        "daemon_pinned_d2h_with_staging_e2",
        "daemon_pinned_d2h_with_checksum",
        "full_csd_put",
    ]
    for mode in modes:
        if mode == "local_pinned_d2h_only":
            mode_local_pinned_d2h_only(args)
        elif mode == "daemon_pinned_d2h_no_ipc":
            mode_daemon_pinned_d2h_no_ipc(args)
        elif mode == "daemon_pinned_d2h_with_ipc_cached":
            mode_ipc_cached_or_open(args, mode, per_chunk_open=False)
        elif mode == "daemon_pinned_d2h_with_ipc_open_per_chunk":
            mode_ipc_cached_or_open(args, mode, per_chunk_open=True)
        elif mode == "daemon_pinned_d2h_with_staging_e1":
            mode_staging(args, mode, reuse_slab=False)
        elif mode == "daemon_pinned_d2h_with_staging_e2":
            mode_staging(args, mode, reuse_slab=True)
        elif mode == "daemon_pinned_d2h_with_checksum":
            mode_ipc_cached_or_open(args, mode, per_chunk_open=False, checksum=True)
        elif mode == "full_csd_put":
            mode_full_csd_put(args)
        else:
            raise ValueError(f"unknown mode {mode}")


def main() -> None:
    global SUMMARY_ONLY
    parser = argparse.ArgumentParser(description="Layered RACER CSD native pinned benchmark")
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "local_pinned_d2h_only",
            "daemon_pinned_d2h_no_ipc",
            "daemon_pinned_d2h_with_ipc_cached",
            "daemon_pinned_d2h_with_ipc_open_per_chunk",
            "daemon_pinned_d2h_with_staging_e1",
            "daemon_pinned_d2h_with_staging_e2",
            "daemon_pinned_d2h_with_checksum",
            "full_csd_put",
        ],
        default="all",
    )
    parser.add_argument("--num-chunks", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--checksum-type", choices=["none", "sample64", "sha256", "crc32", "xxh64"], default="none")
    parser.add_argument("--manifest-update-mode", choices=["batch", "per_chunk"], default="batch")
    parser.add_argument("--csd-total-bytes", type=int, default=None)
    parser.add_argument("--include-sqlite", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    SUMMARY_ONLY = bool(args.summary_only)
    run(args)


if __name__ == "__main__":
    main()
