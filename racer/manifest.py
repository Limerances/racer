"""Manifest construction and state-dict metadata conversion."""

from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any

import torch

from . import checksum, routing
from .config import RacerConfig
from .layout import ElasticLayout
from .state_dict_codec import NonTensorMetadata, RankStateMetadata, TensorMetadata
from .storage import StoredCheckpoint, StoredReductionGroup


def state_metadata_to_manifest(metadata: RankStateMetadata) -> dict[str, Any]:
    return {
        "source_train_rank": int(metadata.source_train_rank),
        "payload_nbytes": int(metadata.payload_nbytes),
        "tensors": [
            {
                "key": item.key,
                "dtype": item.dtype,
                "shape": list(item.shape),
                "device": item.device,
                "requires_grad": bool(item.requires_grad),
                "was_contiguous": bool(item.was_contiguous),
                "offset": int(item.offset),
                "nbytes": int(item.nbytes),
            }
            for item in metadata.tensors
        ],
        "non_tensors": [
            {
                "key": item.key,
                "value": item.value,
            }
            for item in getattr(metadata, "non_tensors", [])
        ],
    }


def state_metadata_from_manifest(data: dict[str, Any]) -> RankStateMetadata:
    tensors = [
        TensorMetadata(
            key=str(item["key"]),
            dtype=str(item["dtype"]),
            shape=tuple(int(dim) for dim in item["shape"]),
            device=str(item["device"]),
            requires_grad=bool(item["requires_grad"]),
            was_contiguous=bool(item["was_contiguous"]),
            offset=int(item["offset"]),
            nbytes=int(item["nbytes"]),
        )
        for item in data.get("tensors", [])
    ]
    return RankStateMetadata(
        source_train_rank=int(data["source_train_rank"]),
        tensors=tensors,
        payload_nbytes=int(data["payload_nbytes"]),
        non_tensors=[
            NonTensorMetadata(key=str(item["key"]), value=item.get("value"))
            for item in data.get("non_tensors", [])
        ],
    )


def chunk_id(reduction_group_index: int, row: int) -> str:
    return f"rg_{reduction_group_index:06d}_row_{row:03d}"


def build_manifest(
    *,
    tag: str,
    config: RacerConfig,
    matrix: list[list[int]],
    elastic_layout: ElasticLayout,
    stored_reduction_groups: list[StoredReductionGroup],
    plan: routing.RoutingPlan,
    host_buffer_owner_rank,
    checksum_fn,
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    virtual_slots: list[dict[str, Any]] = []
    for group in elastic_layout.reduction_groups:
        for slot in group:
            if slot.is_virtual_zero:
                virtual_slots.append(
                    {
                        "slot_id": slot.slot_id,
                        "data_group_id": slot.data_group_id,
                        "relative_index": slot.relative_index,
                        "train_rank": None,
                        "is_virtual_zero": True,
                        "valid_nbytes": slot.valid_nbytes,
                    }
                )

    for reduction_group in stored_reduction_groups:
        group = elastic_layout.reduction_groups[reduction_group.index]
        slot_entries = [
            {
                "slot_id": slot.slot_id,
                "data_group_id": slot.data_group_id,
                "relative_index": slot.relative_index,
                "train_rank": slot.train_rank,
                "is_virtual_zero": slot.is_virtual_zero,
                "valid_nbytes": reduction_group.numels.get(slot.train_rank, 0) if slot.train_rank is not None else 0,
                "shape": list(reduction_group.shapes.get(slot.train_rank, ())) if slot.train_rank is not None else [],
            }
            for slot in group
        ]
        for row, tensor in enumerate(reduction_group.rows):
            owner = int(config.train_ranks[row])
            current_chunk_id = chunk_id(reduction_group.index, row)
            host_owner = host_buffer_owner_rank(reduction_group.data_ranks, row)
            chunks.append(
                {
                    "chunk_id": current_chunk_id,
                    "reduction_group_index": reduction_group.index,
                    "row": row,
                    "chunk_role": "data" if row < config.k else "parity",
                    "parity_id": None if row < config.k else row - config.k,
                    "owner_rank": owner,
                    "host_buffer_owner_rank": host_owner,
                    "host_buffer_role": "train_local" if row < config.k else "spare_local",
                    "is_spare_owned": False,
                    "stored_device": str(tensor.device),
                    "is_pinned_host": False,
                    "num_bytes": int(tensor.numel()),
                    "checksum": checksum_fn(tensor),
                    "slots": slot_entries,
                }
            )

    return {
        "tag": tag,
        "version": 1,
        "k": config.k,
        "m": config.m,
        "n_train": len(config.train_ranks),
        "E": [row[:] for row in matrix],
        "train_ranks": list(config.train_ranks),
        "spare_ranks": list(config.spare_ranks),
        "routing_plan": asdict(plan),
        "routing_cost": asdict(plan.cost),
        "elastic_layout": {
            "q": elastic_layout.q,
            "virtual_W": elastic_layout.virtual_W,
            "num_virtual_zero": elastic_layout.num_virtual_zero,
        },
        "virtual_slots": virtual_slots,
        "chunks": chunks,
        "chunk_owner": {chunk["chunk_id"]: chunk["owner_rank"] for chunk in chunks},
        "checksum": checksum.manifest_checksum(chunks),
    }


def write_chunks_and_manifest(
    *,
    chunk_storage: Any,
    tag: str,
    stored_reduction_groups: list[StoredReductionGroup],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    by_id = {chunk["chunk_id"]: chunk for chunk in manifest["chunks"]}
    use_cuda_ipc = False
    strict_daemon_storage = False
    if hasattr(chunk_storage, "put_cuda_tensor") and hasattr(chunk_storage, "wait"):
        capabilities_fn = getattr(chunk_storage, "capabilities", None)
        if capabilities_fn is not None:
            caps = dict(capabilities_fn())
            strict_daemon_storage = bool(caps.get("restart_aware", False)) and bool(caps.get("daemon_owned", False))
            use_cuda_ipc = bool(caps.get("supports_cuda_ipc")) and bool(caps.get("supports_async_copy"))
    if not strict_daemon_storage:
        raise RuntimeError(
            "RACER store requires restart-aware daemon-owned storage; "
            "in-process and compatibility storage paths are disabled"
        )
    if not use_cuda_ipc:
        raise RuntimeError(
            "restart-aware daemon storage requires CUDA IPC async copy; no fallback put path is allowed"
        )
    metrics: dict[str, Any] = {
        "data_chunk_write_ms": 0.0,
        "parity_chunk_write_ms": 0.0,
        "manifest_write_ms": 0.0,
        "data_chunk_bytes": 0,
        "parity_chunk_bytes": 0,
        "data_chunk_count": 0,
        "parity_chunk_count": 0,
    }
    if hasattr(chunk_storage, "begin"):
        start = time.perf_counter()
        chunk_storage.begin(tag, manifest_base=manifest)
        metrics["begin_ms"] = (time.perf_counter() - start) * 1000.0
    for reduction_group in stored_reduction_groups:
        for row, tensor in enumerate(reduction_group.rows):
            current_chunk_id = chunk_id(reduction_group.index, row)
            role = "data" if row < len(reduction_group.data_ranks) else "parity"
            start = time.perf_counter()
            if use_cuda_ipc and tensor.device.type == "cuda":
                op_id = chunk_storage.put_cuda_tensor(tag, current_chunk_id, tensor, by_id[current_chunk_id])
                metrics.setdefault("_async_futures", []).append((str(op_id), tensor, role, start))
                elapsed_ms = (time.perf_counter() - start) * 1000.0
            else:
                raise RuntimeError(
                    f"chunk {current_chunk_id} is not CUDA-backed for daemon storage; no CPU fallback is allowed"
                )
            metrics[f"{role}_chunk_write_ms"] += elapsed_ms
            metrics[f"{role}_chunk_bytes"] += int(tensor.numel())
            metrics[f"{role}_chunk_count"] += 1
    wait_start = time.perf_counter()
    async_profile: dict[str, float] = {}
    async_futures = metrics.pop("_async_futures", [])
    for op_id, _tensor, _role, _enqueue_start in async_futures:
        result = chunk_storage.wait(op_id)
        if isinstance(result, dict):
            profile = result.get("profile")
            if isinstance(profile, dict):
                for key, value in profile.items():
                    if isinstance(value, (int, float)):
                        async_profile[f"csd_{key}"] = async_profile.get(f"csd_{key}", 0.0) + float(value)
    if async_futures:
        metrics["async_chunk_wait_ms"] = (time.perf_counter() - wait_start) * 1000.0
        metrics.update(async_profile)
    start = time.perf_counter()
    chunk_storage.put_manifest(tag, manifest)
    metrics["manifest_write_ms"] = (time.perf_counter() - start) * 1000.0
    if hasattr(chunk_storage, "commit"):
        start = time.perf_counter()
        chunk_storage.commit(tag)
        metrics["commit_ms"] = (time.perf_counter() - start) * 1000.0
    return metrics


def validate_committed_daemon_manifest(manifest: dict[str, Any], *, tag: str) -> None:
    missing = [
        key
        for key in ("committed", "daemon_owned", "data_resident")
        if not bool(manifest.get(key, False))
    ]
    if missing:
        raise RuntimeError(
            f"RACER checkpoint {tag!r} is not a committed daemon-resident checkpoint; "
            f"missing/false flags: {', '.join(missing)}"
        )


def checkpoint_from_chunk_storage(
    *,
    tag: str,
    chunk_storage: Any,
) -> StoredCheckpoint:
    manifest = chunk_storage.get_manifest(tag)
    validate_committed_daemon_manifest(manifest, tag=tag)
    use_cuda_ipc_read = False
    strict_daemon_storage = False
    if hasattr(chunk_storage, "read_into_cuda_tensor") and hasattr(chunk_storage, "wait"):
        capabilities_fn = getattr(chunk_storage, "capabilities", None)
        if capabilities_fn is not None:
            caps = dict(capabilities_fn())
            strict_daemon_storage = bool(caps.get("restart_aware", False)) and bool(caps.get("daemon_owned", False))
            use_cuda_ipc_read = bool(caps.get("supports_cuda_ipc")) and bool(caps.get("supports_async_copy", False))
    if not strict_daemon_storage:
        raise RuntimeError(
            "RACER load requires restart-aware daemon-owned storage; "
            "in-process and compatibility storage paths are disabled"
        )
    if not use_cuda_ipc_read:
        raise RuntimeError(
            "restart-aware daemon storage requires CUDA IPC async reads; no fallback get path is allowed"
        )
    reduction_groups: list[StoredReductionGroup] = []
    chunks_by_reduction_group: dict[int, list[dict[str, Any]]] = {}
    for chunk in manifest["chunks"]:
        chunks_by_reduction_group.setdefault(int(chunk["reduction_group_index"]), []).append(chunk)
    for reduction_group_index in sorted(chunks_by_reduction_group):
        chunk_entries = sorted(chunks_by_reduction_group[reduction_group_index], key=lambda item: int(item["row"]))
        rows: list[torch.Tensor] = []
        read_futures: list[tuple[int, str, torch.Tensor]] = []
        data_ranks: list[int | None] = [None] * int(manifest["k"])
        shapes: dict[int, tuple[int, ...]] = {}
        numels: dict[int, int] = {}
        for chunk in chunk_entries:
            chunk_id_value = str(chunk["chunk_id"])
            stored_device = torch.device(str(chunk.get("stored_device", "cpu")))
            if use_cuda_ipc_read and stored_device.type != "cuda" and chunk.get("owner_rank") is not None:
                stored_device = torch.device("cuda", int(chunk["owner_rank"]))
            nbytes = int(chunk.get("nbytes", chunk.get("num_bytes", 0)) or 0)
            if use_cuda_ipc_read and stored_device.type == "cuda":
                dst = torch.empty(nbytes, dtype=torch.uint8, device=stored_device)
                op_id = chunk_storage.read_into_cuda_tensor(tag, chunk_id_value, dst)
                rows.append(dst)
                read_futures.append((len(rows) - 1, str(op_id), dst))
            else:
                raise RuntimeError(
                    f"chunk {chunk_id_value} cannot be read with CUDA IPC; no CPU/socket fallback is allowed"
                )
            for slot in chunk.get("slots", []):
                data_group_id = int(slot["data_group_id"])
                if data_group_id < 0 or data_group_id >= len(data_ranks):
                    continue
                rank = slot.get("train_rank")
                if rank is None:
                    continue
                rank = int(rank)
                data_ranks[data_group_id] = rank
                shape = tuple(int(v) for v in slot.get("shape", []))
                shapes[rank] = shape if shape else (int(slot["valid_nbytes"]),)
                numels[rank] = int(slot["valid_nbytes"])
        for _row_index, op_id, _dst in read_futures:
            chunk_storage.wait(op_id)
        reduction_groups.append(
            StoredReductionGroup(
                index=reduction_group_index,
                rows=rows,
                data_ranks=tuple(data_ranks),
                reduction_group_bytes=max((int(row.numel()) for row in rows), default=0),
                shapes=shapes,
                numels=numels,
            )
        )
    return StoredCheckpoint(tag=tag, reduction_groups=reduction_groups, matrix=manifest["E"], metadata=manifest)
