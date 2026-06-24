#!/usr/bin/env python3
"""Benchmark RACER CSD storage backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch

import racer
from racer.csd import CheckpointStorageDaemonClient


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def _make_chunk(nbytes: int, device: torch.device, seed: int) -> torch.Tensor:
    cpu = (torch.arange(int(nbytes), dtype=torch.int64) + int(seed)).remainder(251).to(torch.uint8)
    return cpu.to(device) if device.type == "cuda" else cpu


def _wait_one(client: CheckpointStorageDaemonClient, item) -> float:
    op_id, submit_ts, _tensor = item
    client.wait(op_id)
    return (time.perf_counter() - submit_ts) * 1000.0


def _put_chunks(
    client: CheckpointStorageDaemonClient,
    *,
    tag: str,
    backend_caps: dict,
    chunks: list[torch.Tensor],
    pipeline_depth: int,
) -> list[float]:
    latencies: list[float] = []
    pending = []
    use_native = bool(backend_caps.get("supports_cuda_ipc")) and chunks and chunks[0].device.type == "cuda"
    for index, tensor in enumerate(chunks):
        metadata = {"row": index, "owner_rank": 0, "writer_rank": 0, "nbytes": int(tensor.numel())}
        start = time.perf_counter()
        if not use_native:
            raise RuntimeError("CSD benchmark requires CUDA IPC native transport; no put fallback is allowed")
        op_id = client.put_cuda_tensor(tag, f"c{index}", tensor, metadata)
        pending.append((op_id, start, tensor))
        if len(pending) >= int(pipeline_depth):
            latencies.append(_wait_one(client, pending.pop(0)))
    while pending:
        latencies.append(_wait_one(client, pending.pop(0)))
    return latencies


def _get_chunks(
    client: CheckpointStorageDaemonClient,
    *,
    tag: str,
    backend_caps: dict,
    num_chunks: int,
    chunk_nbytes: int,
    device: torch.device,
    pipeline_depth: int,
) -> list[float]:
    latencies: list[float] = []
    pending = []
    use_native = bool(backend_caps.get("supports_cuda_ipc")) and device.type == "cuda"
    for index in range(int(num_chunks)):
        start = time.perf_counter()
        if not use_native:
            raise RuntimeError("CSD benchmark requires CUDA IPC native transport; no get fallback is allowed")
        dst = torch.empty(int(chunk_nbytes), dtype=torch.uint8, device=device)
        op_id = client.read_into_cuda_tensor(tag, f"c{index}", dst)
        pending.append((op_id, start, dst))
        if len(pending) >= int(pipeline_depth):
            latencies.append(_wait_one(client, pending.pop(0)))
    while pending:
        latencies.append(_wait_one(client, pending.pop(0)))
    return latencies


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RACER Checkpoint Storage Daemon")
    parser.add_argument("--backend", choices=["native_pinned"], default="native_pinned")
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pipeline-depth", type=int, default=2)
    parser.add_argument("--direction", choices=["put", "get", "both"], default="both")
    args = parser.parse_args()

    chunk_nbytes = int(args.chunk_mib) * 1024 * 1024
    if args.backend == "native_pinned" and not torch.cuda.is_available():
        raise SystemExit("native_pinned benchmark requires CUDA")
    device = torch.device("cuda", int(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    with tempfile.TemporaryDirectory(prefix="racer-csd-bench-") as tmp:
        tmp_path = Path(tmp)
        socket_path = tmp_path / "csd.sock"
        backend_options = {
            "total_bytes": int(args.num_chunks) * int(chunk_nbytes),
            "segment_bytes": max(chunk_nbytes, 256 * 1024 * 1024),
            "device": int(args.device),
        }
        daemon = racer.start_checkpoint_storage_daemon(
            socket_path=socket_path,
            metadata_dir=tmp_path / "metadata",
            backend=args.backend,
            backend_options=backend_options,
        )
        try:
            client = daemon.client
            caps = client.capabilities()
            tag = "bench"
            manifest = {
                "tag": tag,
                "k": 1,
                "m": 0,
                "train_ranks": [0],
                "spare_ranks": [],
                "chunks": [
                    {"chunk_id": f"c{index}", "row": index, "owner_rank": 0, "num_bytes": chunk_nbytes}
                    for index in range(int(args.num_chunks))
                ],
            }
            client.begin(tag, manifest, expected_chunks=int(args.num_chunks))
            total_bytes = int(args.num_chunks) * int(chunk_nbytes)
            source_chunks = [
                _make_chunk(chunk_nbytes, device, index)
                for index in range(int(args.num_chunks))
            ]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_start = time.perf_counter()
            latencies: list[float] = []
            if args.direction in {"put", "both"}:
                latencies.extend(
                    _put_chunks(
                        client,
                        tag=tag,
                        backend_caps=caps,
                        chunks=source_chunks,
                        pipeline_depth=max(1, int(args.pipeline_depth)),
                    )
                )
                client.put_manifest(tag, manifest)
                client.commit(tag)
            else:
                _put_chunks(
                    client,
                    tag=tag,
                    backend_caps=caps,
                    chunks=source_chunks,
                    pipeline_depth=max(1, int(args.pipeline_depth)),
                )
                client.put_manifest(tag, manifest)
                client.commit(tag)
            if args.direction in {"get", "both"}:
                latencies.extend(
                    _get_chunks(
                        client,
                        tag=tag,
                        backend_caps=caps,
                        num_chunks=int(args.num_chunks),
                        chunk_nbytes=chunk_nbytes,
                        device=device,
                        pipeline_depth=max(1, int(args.pipeline_depth)),
                    )
                )
            elapsed_ms = (time.perf_counter() - elapsed_start) * 1000.0
            direction_multiplier = 2 if args.direction == "both" else 1
            effective_gib_s = (total_bytes * direction_multiplier / (1024**3)) / max(elapsed_ms / 1000.0, 1e-9)
            result = {
                "backend": args.backend,
                "num_chunks": int(args.num_chunks),
                "chunk_mib": int(args.chunk_mib),
                "total_gib": total_bytes * direction_multiplier / (1024**3),
                "elapsed_ms": elapsed_ms,
                "effective_gib_s": effective_gib_s,
                "p50_chunk_ms": statistics.median(latencies) if latencies else 0.0,
                "p95_chunk_ms": _percentile(latencies, 95.0),
                "max_chunk_ms": max(latencies) if latencies else 0.0,
                "pipeline_depth": int(args.pipeline_depth),
                "cuda_native_pinned": bool(caps.get("cuda_native_pinned")),
                "async_copy": bool(caps.get("supports_async_copy")),
            }
            print(json.dumps(result, indent=2, sort_keys=True))
        finally:
            daemon.shutdown()


if __name__ == "__main__":
    main()
