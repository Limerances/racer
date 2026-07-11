"""Distributed RACER executor for torchrun spare-GPU prototypes.

This module implements the NCCL spare-compute path only: train ranks send raw
CUDA byte buffers to the spare GPU, the spare GPU computes parity with RACER
CUDA GF kernels, and parity rows are returned to train-rank chunk owners.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from . import cauchy, codec_cuda, routing
from .config import RacerConfig
from .layout import ElasticLayout
from .manifest import validate_committed_daemon_manifest
from .routing import RoutingPlan


@dataclass
class DistributedStoreResult:
    tag: str
    config: RacerConfig
    layout: ElasticLayout
    matrix: list[list[int]]
    plan: RoutingPlan
    local_chunks: dict[str, torch.Tensor]
    packet_nbytes_by_rank: dict[int, int]
    manifest: dict[str, Any] | None = None
    profile: dict[str, Any] | None = None
    storage_backed: bool = False

    @property
    def routing_cost(self):
        return self.plan.cost


@dataclass
class DistributedStorePendingPut:
    tag: str
    chunk_id: str
    op_id: str
    tensor: torch.Tensor


@dataclass
class DistributedStoreBatchHandle:
    states: list[DistributedStoreResult]
    chunk_storage: Any
    rank: int
    pending_puts_by_tag: list[list[DistributedStorePendingPut]]
    retained_tensors_by_tag: list[list[torch.Tensor]]
    storage_profiles: list[dict[str, float]]
    execute_ms_by_tag: list[float]
    batch_start: float
    prepare_total_ms: float
    finalize_started: bool = False
    finalized: bool = False

    @property
    def tags(self) -> list[str]:
        return [state.tag for state in self.states]


@dataclass
class DistributedLoadResult:
    recovered: dict[int, torch.Tensor]
    decode_rank: int
    profile: dict[str, Any] | None = None


def is_distributed_available() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def get_rank_or_zero() -> int:
    if not is_distributed_available():
        return 0
    return int(dist.get_rank())


def _require_dist() -> None:
    if not is_distributed_available():
        raise RuntimeError("torch.distributed must be initialized before using racer.distributed")


def _rank(process_group: Any | None = None) -> int:
    if process_group is not None:
        return int(process_group.rank())
    _require_dist()
    return int(dist.get_rank())


def _world_size(process_group: Any | None = None) -> int:
    if process_group is not None:
        return int(process_group.size())
    _require_dist()
    return int(dist.get_world_size())


def _current_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed RACER")
    return torch.device("cuda", torch.cuda.current_device())


def _require_nccl(process_group: Any | None = None) -> None:
    if process_group is not None:
        return
    _require_dist()
    if str(dist.get_backend()) != "nccl":
        raise RuntimeError("distributed RACER spare_compute requires the default process group backend to be NCCL")


def _barrier(process_group: Any | None = None) -> None:
    if process_group is None:
        dist.barrier()
    else:
        process_group.barrier().wait()


def _stream_barrier(process_group: Any) -> None:
    """Order runtime-PG work on the current CUDA stream without CPU waiting.

    ProcessGroupNCCL treats barrier().wait() specially and synchronizes the
    CPU thread. synchronize() only installs the CUDA stream dependency,
    retaining the P2P ordering boundary without defeating asynchronous launch.
    """

    if process_group is None:
        raise RuntimeError("RACER stream barrier requires an explicit process group")
    work = process_group.barrier()
    work.synchronize()


def _chunk_id(reduction_group_index: int, row: int) -> str:
    return f"rg_{reduction_group_index:06d}_row_{row:03d}"


def _build_storage_manifest(
    *,
    tag: str,
    config: RacerConfig,
    layout: ElasticLayout,
    matrix: list[list[int]],
    plan: RoutingPlan,
    group_nbytes: dict[int, int],
    packet_sizes: dict[int, int],
) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    for group in layout.reduction_groups:
        reduction_group_index = int(group[0].relative_index)
        nbytes = int(group_nbytes.get(reduction_group_index, 0))
        slot_entries = [
            {
                "slot_id": int(slot.slot_id),
                "data_group_id": int(slot.data_group_id),
                "relative_index": int(slot.relative_index),
                "train_rank": None if slot.train_rank is None else int(slot.train_rank),
                "is_virtual_zero": bool(slot.is_virtual_zero),
                "valid_nbytes": (
                    int(packet_sizes.get(int(slot.train_rank), 0))
                    if slot.train_rank is not None
                    else 0
                ),
            }
            for slot in group
        ]
        for row, owner in enumerate(config.train_ranks):
            chunks.append(
                {
                    "chunk_id": _chunk_id(reduction_group_index, row),
                    "reduction_group_index": reduction_group_index,
                    "row": int(row),
                    "chunk_role": "data" if row < config.k else "parity",
                    "parity_id": None if row < config.k else int(row - config.k),
                    "owner_rank": int(owner),
                    "num_bytes": nbytes,
                    "slots": slot_entries,
                }
            )
    return {
        "tag": str(tag),
        "version": 1,
        "kind": "racer_distributed_store",
        "k": int(config.k),
        "m": int(config.m),
        "train_ranks": list(config.train_ranks),
        "spare_ranks": list(config.spare_ranks),
        "E": [list(row) for row in matrix],
        "routing_plan": asdict(plan),
        "elastic_layout": {
            "q": int(layout.q),
            "virtual_W": int(layout.virtual_W),
            "num_virtual_zero": int(layout.num_virtual_zero),
        },
        "packet_nbytes_by_rank": {int(rank): int(nbytes) for rank, nbytes in packet_sizes.items()},
        "group_nbytes": {int(group): int(nbytes) for group, nbytes in group_nbytes.items()},
        "chunks": chunks,
        "chunk_owner": {chunk["chunk_id"]: int(chunk["owner_rank"]) for chunk in chunks},
    }


def _zero_data_chunk_ids(manifest: dict[str, Any] | None) -> set[str]:
    if not manifest:
        return set()
    zero_chunks: set[str] = set()
    for chunk in manifest.get("chunks", []):
        if str(chunk.get("chunk_role")) != "data":
            continue
        row = int(chunk.get("row", -1))
        for slot in chunk.get("slots", []):
            if int(slot.get("data_group_id", -2)) != row:
                continue
            if bool(slot.get("is_virtual_zero", False)) or int(slot.get("valid_nbytes", 0)) == 0:
                zero_chunks.add(str(chunk["chunk_id"]))
            break
    return zero_chunks


def _storage_read_checksum_debug_enabled() -> bool:
    return any(
        str(os.environ.get(name, "")).lower() not in {"", "0", "false", "no", "off", "none", "disabled"}
        for name in ("RACER_DEBUG_STORAGE_READ_CHECKSUM", "RACER_DEBUG_PAYLOAD_CHECKSUM")
    )


def _storage_read_checksum_mode(expected: str, checksum_type: str) -> str | None:
    normalized_type = str(checksum_type or "").lower().replace("-", "_")
    if expected.startswith(("sample64-v1:", "sum64-v1:")):
        return "fast"
    if expected.startswith("sha256-v1:"):
        return "sha256"
    if normalized_type in {"sample64", "sample64_v1", "fast", "sampled"}:
        return "fast"
    if normalized_type in {"sha256", "sha256_v1", "strict", "strict_sha256"}:
        return "sha256"
    return None


def _debug_storage_read_checksum(
    *,
    tag: str,
    rank: int,
    chunk_id: str,
    chunk: Mapping[str, Any],
    tensor: torch.Tensor,
) -> tuple[int, int, float]:
    if not _storage_read_checksum_debug_enabled():
        return 0, 0, 0.0
    expected = str(chunk.get("checksum", "") or "")
    checksum_type = str(chunk.get("checksum_type", "") or "")
    mode = _storage_read_checksum_mode(expected, checksum_type)
    if not expected or mode is None:
        print(
            f"RACER storage read checksum skipped: rank={rank}, tag={tag}, "
            f"chunk_id={chunk_id}, checksum_type={checksum_type!r}, expected_present={bool(expected)}",
            flush=True,
        )
        return 0, 0, 0.0
    nbytes = int(chunk.get("nbytes", chunk.get("num_bytes", int(tensor.numel()))) or 0)
    nbytes = min(nbytes, int(tensor.numel()))
    view = tensor.narrow(0, 0, nbytes)

    def sync_device(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    from .checksum import tensor_checksum

    start = time.perf_counter()
    try:
        actual = tensor_checksum(
            view,
            buffer_size=16 * 1024 * 1024,
            sync_device=sync_device,
            mode=mode,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print(
            f"RACER storage read checksum error: rank={rank}, tag={tag}, chunk_id={chunk_id}, "
            f"checksum_type={checksum_type!r}, error={type(exc).__name__}: {exc}",
            flush=True,
        )
        return 0, 1, elapsed_ms
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if actual != expected:
        print(
            f"RACER storage read checksum mismatch: rank={rank}, tag={tag}, chunk_id={chunk_id}, "
            f"expected={expected}, got={actual}, nbytes={nbytes}",
            flush=True,
        )
        return 0, 1, elapsed_ms
    print(
        f"RACER storage read checksum verified: rank={rank}, tag={tag}, chunk_id={chunk_id}, "
        f"checksum={actual}, nbytes={nbytes}",
        flush=True,
    )
    return 1, 0, elapsed_ms


def _expected_storage_chunk_count(manifest: dict[str, Any] | None) -> int:
    if not manifest:
        return 0
    zero_chunks = _zero_data_chunk_ids(manifest)
    return sum(1 for chunk in manifest.get("chunks", []) if str(chunk.get("chunk_id")) not in zero_chunks)


def _csv_ints(value: str | None) -> list[int]:
    if not value:
        return []
    out: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(item))
    return out


def _align_nbytes(value: int, alignment: int) -> int:
    value = int(value)
    alignment = max(1, int(alignment))
    return (value + alignment - 1) // alignment * alignment


def _distributed_group_alignment_bytes() -> int:
    value = os.environ.get("RACER_DISTRIBUTED_GROUP_ALIGNMENT_BYTES", "2097152")
    if value in (None, ""):
        return 4096
    return max(1, int(value))


def _per_node_csd_enabled() -> bool:
    return os.environ.get("RACER_CSD_PER_NODE", "0").lower() in {"1", "true", "yes", "on"}


def _local_csd_owner_ranks(default_rank: int) -> set[int]:
    ranks = _csv_ints(os.environ.get("RACER_CSD_LOCAL_RANKS"))
    return set(ranks if ranks else [int(default_rank)])


def _local_csd_coordinator_rank(default_rank: int) -> int:
    value = os.environ.get("RACER_CSD_LOCAL_COORDINATOR_RANK")
    if value not in (None, ""):
        return int(value)
    return min(_local_csd_owner_ranks(default_rank))


def _expected_storage_chunk_count_for_ranks(manifest: dict[str, Any] | None, owner_ranks: set[int]) -> int:
    if not manifest:
        return 0
    zero_chunks = _zero_data_chunk_ids(manifest)
    return sum(
        1
        for chunk in manifest.get("chunks", [])
        if str(chunk.get("chunk_id")) not in zero_chunks and int(chunk.get("owner_rank", -1)) in owner_ranks
    )


def _storage_supports_cuda_ipc(chunk_storage: Any) -> bool:
    if not hasattr(chunk_storage, "put_cuda_tensor") or not hasattr(chunk_storage, "wait"):
        return False
    capabilities_fn = getattr(chunk_storage, "capabilities", None)
    if capabilities_fn is None:
        return False
    try:
        caps = dict(capabilities_fn())
    except Exception:
        return False
    return bool(caps.get("supports_cuda_ipc")) and bool(caps.get("supports_async_copy"))


def _accumulate_profile(target: dict[str, float], source: Mapping[str, float]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0.0) + float(value)


def _begin_storage_checkpoint(
    *,
    chunk_storage: Any,
    state: DistributedStoreResult,
    rank: int,
    process_group: Any | None = None,
) -> dict[str, float]:
    coordinator = int(state.config.train_ranks[0])
    storage_profile: dict[str, float] = {
        "storage_begin_ms": 0.0,
        "storage_begin_barrier_ms": 0.0,
    }
    if _per_node_csd_enabled():
        local_owner_ranks = _local_csd_owner_ranks(rank)
        should_begin = rank == _local_csd_coordinator_rank(rank)
        expected_chunks = _expected_storage_chunk_count_for_ranks(state.manifest, local_owner_ranks)
    else:
        should_begin = rank == coordinator
        expected_chunks = _expected_storage_chunk_count(state.manifest)
    if should_begin:
        begin_start = time.perf_counter()
        chunk_storage.begin(
            state.tag,
            manifest_base=state.manifest or {},
            expected_chunks=expected_chunks,
        )
        storage_profile["storage_begin_ms"] = (time.perf_counter() - begin_start) * 1000.0
    begin_barrier_start = time.perf_counter()
    _barrier(process_group)
    storage_profile["storage_begin_barrier_ms"] = (time.perf_counter() - begin_barrier_start) * 1000.0
    return storage_profile


def _enqueue_storage_chunks(
    *,
    chunk_storage: Any,
    state: DistributedStoreResult,
    chunks: Mapping[str, torch.Tensor],
    rank: int,
) -> tuple[int, int, dict[str, float], list[DistributedStorePendingPut]]:
    zero_chunks = _zero_data_chunk_ids(state.manifest)
    storage_profile: dict[str, float] = {
        "storage_enqueue_ms": 0.0,
    }
    stored_nbytes = 0
    stored_count = 0
    pending: list[DistributedStorePendingPut] = []
    use_cuda_ipc = _storage_supports_cuda_ipc(chunk_storage)
    enqueue_start = time.perf_counter()
    for chunk_id, chunk in chunks.items():
        if chunk_id in zero_chunks:
            continue
        metadata = {"owner_rank": int(rank), "writer_rank": int(rank), "nbytes": int(chunk.numel())}
        checksum_type = os.environ.get("RACER_CSD_CHECKSUM_TYPE")
        if checksum_type:
            metadata["checksum_type"] = str(checksum_type)
        manifest_update_mode = os.environ.get("RACER_CSD_MANIFEST_UPDATE_MODE")
        if manifest_update_mode:
            metadata["manifest_update_mode"] = str(manifest_update_mode)
        if use_cuda_ipc and chunk.device.type == "cuda":
            op_id = chunk_storage.put_cuda_tensor(state.tag, chunk_id, chunk, metadata)
            pending.append(
                DistributedStorePendingPut(
                    tag=state.tag,
                    chunk_id=str(chunk_id),
                    op_id=str(op_id),
                    tensor=chunk,
                )
            )
        else:
            raise RuntimeError(
                "RACER distributed store requires daemon storage with CUDA IPC async write support; "
                "CPU/socket fallback put is disabled"
            )
        stored_nbytes += int(chunk.numel())
        stored_count += 1
    storage_profile["storage_enqueue_ms"] = (time.perf_counter() - enqueue_start) * 1000.0
    return stored_nbytes, stored_count, storage_profile, pending


def _wait_storage_puts(
    *,
    chunk_storage: Any,
    pending: Sequence[DistributedStorePendingPut],
) -> dict[str, float]:
    storage_profile: dict[str, float] = {"storage_wait_ms": 0.0}
    wait_start = time.perf_counter()
    for put in pending:
        result = chunk_storage.wait(put.op_id)
        if isinstance(result, dict):
            profile = result.get("profile")
            if isinstance(profile, dict):
                for key, value in profile.items():
                    if isinstance(value, (int, float)):
                        storage_profile[f"csd_{key}"] = storage_profile.get(f"csd_{key}", 0.0) + float(value)
                source = profile.get("daemon_allocate_source")
                if isinstance(source, str) and source:
                    count_key = f"csd_daemon_allocate_source_{source}_count"
                    storage_profile[count_key] = storage_profile.get(count_key, 0.0) + 1.0
    storage_profile["storage_wait_ms"] = (time.perf_counter() - wait_start) * 1000.0
    return storage_profile


def _put_storage_chunks(
    *,
    chunk_storage: Any,
    state: DistributedStoreResult,
    chunks: Mapping[str, torch.Tensor],
    rank: int,
) -> tuple[int, int, dict[str, float]]:
    stored_nbytes, stored_count, storage_profile, pending = _enqueue_storage_chunks(
        chunk_storage=chunk_storage,
        state=state,
        chunks=chunks,
        rank=rank,
    )
    wait_profile = _wait_storage_puts(
        chunk_storage=chunk_storage,
        pending=pending,
    )
    _accumulate_profile(storage_profile, wait_profile)
    return stored_nbytes, stored_count, storage_profile


def _commit_storage_checkpoint(
    *,
    chunk_storage: Any,
    state: DistributedStoreResult,
    rank: int,
    process_group: Any | None = None,
) -> dict[str, float]:
    coordinator = int(state.config.train_ranks[0])
    storage_profile: dict[str, float] = {
        "storage_commit_pre_barrier_ms": 0.0,
        "storage_commit_ms": 0.0,
        "storage_commit_post_barrier_ms": 0.0,
    }
    commit_pre_barrier_start = time.perf_counter()
    _barrier(process_group)
    storage_profile["storage_commit_pre_barrier_ms"] = (time.perf_counter() - commit_pre_barrier_start) * 1000.0
    if _per_node_csd_enabled():
        should_commit = rank == _local_csd_coordinator_rank(rank)
    else:
        should_commit = rank == coordinator
    if should_commit:
        commit_start = time.perf_counter()
        chunk_storage.put_manifest(state.tag, state.manifest or {})
        chunk_storage.commit(state.tag)
        storage_profile["storage_commit_ms"] = (time.perf_counter() - commit_start) * 1000.0
    commit_post_barrier_start = time.perf_counter()
    _barrier(process_group)
    storage_profile["storage_commit_post_barrier_ms"] = (time.perf_counter() - commit_post_barrier_start) * 1000.0
    return storage_profile


def _write_storage_chunks(
    *,
    chunk_storage: Any,
    state: DistributedStoreResult,
    rank: int,
    process_group: Any | None = None,
) -> tuple[int, int, dict[str, float]]:
    storage_profile = _begin_storage_checkpoint(
        chunk_storage=chunk_storage,
        state=state,
        rank=rank,
        process_group=process_group,
    )
    stored_nbytes, stored_count, put_profile = _put_storage_chunks(
        chunk_storage=chunk_storage,
        state=state,
        chunks=state.local_chunks,
        rank=rank,
    )
    _accumulate_profile(storage_profile, put_profile)
    commit_profile = _commit_storage_checkpoint(
        chunk_storage=chunk_storage,
        state=state,
        rank=rank,
        process_group=process_group,
    )
    _accumulate_profile(storage_profile, commit_profile)
    return stored_nbytes, stored_count, storage_profile


def _cuda_payload(local_packet: torch.Tensor | None) -> torch.Tensor | None:
    if local_packet is None:
        return None
    return _flat_uint8_view(local_packet)


def _all_gather_int(value: int, process_group: Any | None = None) -> list[int]:
    device = _current_cuda_device()
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    gathered = [torch.zeros_like(tensor) for _ in range(_world_size(process_group))]
    if process_group is None:
        dist.all_gather(gathered, tensor)
    else:
        process_group.allgather(gathered, tensor).wait()
    return [int(item.item()) for item in gathered]


def _zero_buffer(
    nbytes: int,
    *,
    device: torch.device,
    zero_cache: dict[tuple[int | None, int], torch.Tensor] | None = None,
) -> torch.Tensor:
    nbytes = int(nbytes)
    key = (device.index, nbytes)
    if zero_cache is not None:
        cached = zero_cache.get(key)
        if cached is None:
            cached = torch.zeros(nbytes, dtype=torch.uint8, device=device)
            zero_cache[key] = cached
        return cached
    return torch.zeros(nbytes, dtype=torch.uint8, device=device)


def _flat_uint8_view(payload: torch.Tensor) -> torch.Tensor:
    if payload.dtype != torch.uint8:
        raise TypeError("distributed RACER packets must be torch.uint8")
    if payload.device.type != "cuda":
        raise ValueError("distributed RACER spare_compute requires CUDA local packets")
    if payload.is_contiguous():
        return payload.detach().view(-1)
    return payload.detach().contiguous().view(-1)


def _pad(
    payload: torch.Tensor | None,
    nbytes: int,
    *,
    device: torch.device | None = None,
    zero_cache: dict[tuple[int | None, int], torch.Tensor] | None = None,
) -> torch.Tensor:
    if device is None:
        device = payload.device if payload is not None else _current_cuda_device()
    if device.type != "cuda":
        raise ValueError("distributed RACER buffers must stay on CUDA devices")
    nbytes = int(nbytes)
    if payload is None:
        return _zero_buffer(nbytes, device=device, zero_cache=zero_cache)

    flat = _flat_uint8_view(payload)
    if int(flat.numel()) == nbytes and flat.device == device:
        return flat

    out = _zero_buffer(nbytes, device=device, zero_cache=None)
    out[: min(int(flat.numel()), nbytes)].copy_(flat[:nbytes])
    return out


def _payload_store_slot(
    payload: torch.Tensor,
    nbytes: int,
    *,
    device: torch.device,
    zero_slot: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a full-width store slot without mutating outside ``payload``.

    A short tensor can be a view into storage shared with another packet.  Its
    unused backing bytes are therefore not an owned padding region.  Reuse the
    caller-owned group slot when available; the allocation fallback is kept for
    compatibility with call sites that do not provide one.
    """
    flat = _flat_uint8_view(payload)
    nbytes = int(nbytes)
    if nbytes < 0:
        raise ValueError("store payload slot size must be non-negative")
    if flat.device != device:
        raise ValueError("store payload slot must stay on the current CUDA device")
    if int(flat.numel()) == nbytes:
        return flat
    if int(flat.numel()) > nbytes:
        return flat.narrow(0, 0, nbytes)

    valid = int(flat.numel())
    if zero_slot is not None:
        if zero_slot.device != device:
            raise ValueError("store zero slot must stay on the current CUDA device")
        if zero_slot.dtype is not torch.uint8 or not zero_slot.is_contiguous():
            raise ValueError("store zero slot must be a contiguous torch.uint8 tensor")
        if int(zero_slot.numel()) < nbytes:
            raise ValueError(
                "store zero slot is too small: "
                f"capacity_nbytes={int(zero_slot.numel())}, required_nbytes={nbytes}"
            )
        slot = zero_slot.narrow(0, 0, nbytes)
        if valid > 0:
            slot.narrow(0, 0, valid).copy_(flat)
        slot.narrow(0, valid, nbytes - valid).zero_()
        return slot

    out = torch.empty(nbytes, dtype=torch.uint8, device=device)
    if valid > 0:
        out.narrow(0, 0, valid).copy_(flat)
    out.narrow(0, valid, nbytes - valid).zero_()
    return out


def _send_tensor(tensor: torch.Tensor, dst: int, process_group: Any | None = None) -> None:
    if tensor.device.type != "cuda":
        raise ValueError("NCCL send requires a CUDA tensor")
    tensor = tensor.contiguous()
    if process_group is None:
        dist.send(tensor, dst=int(dst))
    else:
        process_group.send([tensor], int(dst), 0).wait()


def _recv_tensor(
    nbytes: int,
    src: int,
    *,
    device: torch.device | None = None,
    process_group: Any | None = None,
) -> torch.Tensor:
    if device is None:
        device = _current_cuda_device()
    out = torch.empty(int(nbytes), dtype=torch.uint8, device=device)
    if process_group is None:
        dist.recv(out, src=int(src))
    else:
        process_group.recv([out], int(src), 0).wait()
    return out


def _recv_tensor_into(
    dst: torch.Tensor,
    nbytes: int,
    src: int,
    *,
    process_group: Any | None = None,
) -> torch.Tensor:
    if dst.device.type != "cuda":
        raise ValueError("NCCL recv slot must be a CUDA tensor")
    nbytes = int(nbytes)
    if int(dst.numel()) < nbytes:
        raise RuntimeError(
            "strict RACER store receive slot is too small; "
            f"slot_nbytes={int(dst.numel())}, required_nbytes={nbytes}"
        )
    out = dst.narrow(0, 0, nbytes)
    if process_group is None:
        dist.recv(out, src=int(src))
    else:
        process_group.recv([out], int(src), 0).wait()
    return out


def _group_nbytes(layout: ElasticLayout, packet_sizes: dict[int, int]) -> dict[int, int]:
    out: dict[int, int] = {}
    alignment = _distributed_group_alignment_bytes()
    for group in layout.reduction_groups:
        size = 0
        for slot in group:
            if slot.train_rank is not None:
                size = max(size, packet_sizes[int(slot.train_rank)])
        if size > 0 and alignment > 1:
            size = _align_nbytes(size, alignment)
        out[group[0].relative_index] = size
    return out


def _packet_sizes_by_rank(
    config: RacerConfig,
    local_payload: torch.Tensor | None,
    process_group: Any | None = None,
) -> dict[int, int]:
    rank = _rank(process_group)
    local_nbytes = int(local_payload.numel()) if rank in config.train_ranks and local_payload is not None else 0
    gathered = _all_gather_int(local_nbytes, process_group)
    return {int(train_rank): int(gathered[int(train_rank)]) for train_rank in config.train_ranks}


def _store_data_rows_for_group(
    *,
    rank: int,
    config: RacerConfig,
    group: Sequence[Any],
    group_nbytes: dict[int, int],
    local_slot_payload: dict[int, torch.Tensor],
    local_chunks: dict[str, torch.Tensor],
    receive_slot: torch.Tensor | None = None,
    zero_send_slot: torch.Tensor | None = None,
    row_sink: Any | None = None,
    process_group: Any | None = None,
) -> int:
    bytes_sent = 0
    device = _current_cuda_device()
    nbytes = group_nbytes[group[0].relative_index]
    for slot in group:
        owner = int(config.train_ranks[slot.data_group_id])
        chunk_id = _chunk_id(slot.relative_index, slot.data_group_id)
        if slot.is_virtual_zero:
            continue
        assert slot.train_rank is not None
        src_rank = int(slot.train_rank)
        if rank == src_rank:
            payload = _payload_store_slot(
                local_slot_payload[src_rank],
                nbytes,
                device=device,
                zero_slot=zero_send_slot,
            )
            if owner == rank:
                local_chunks[chunk_id] = payload
            else:
                _send_tensor(payload, owner, process_group)
                bytes_sent += nbytes
        elif rank == owner:
            if receive_slot is None:
                raise RuntimeError("strict RACER store requires a preallocated slot4/slot3 receive buffer")
            received = _recv_tensor_into(receive_slot, nbytes, src_rank, process_group=process_group)
            if row_sink is None:
                local_chunks[chunk_id] = received
            else:
                row_sink(chunk_id, received)
    return bytes_sent


def _store_data_rows(
    *,
    rank: int,
    config: RacerConfig,
    layout: ElasticLayout,
    group_nbytes: dict[int, int],
    local_slot_payload: dict[int, torch.Tensor],
    local_chunks: dict[str, torch.Tensor],
    process_group: Any | None = None,
) -> int:
    bytes_sent = 0
    for group in layout.reduction_groups:
        bytes_sent += _store_data_rows_for_group(
            rank=rank,
            config=config,
            group=group,
            group_nbytes=group_nbytes,
            local_slot_payload=local_slot_payload,
            local_chunks=local_chunks,
            receive_slot=None,
            zero_send_slot=None,
            row_sink=None,
            process_group=process_group,
        )
    return bytes_sent


def _store_spare_compute_parity_cuda_for_group(
    *,
    rank: int,
    config: RacerConfig,
    group: Sequence[Any],
    E: Sequence[Sequence[int]],
    group_nbytes: dict[int, int],
    local_slot_payload: dict[int, torch.Tensor],
    local_chunks: dict[str, torch.Tensor],
    receive_slot: torch.Tensor | None = None,
    zero_send_slot: torch.Tensor | None = None,
    row_sink: Any | None = None,
    process_group: Any | None = None,
) -> int:
    bytes_sent = 0
    device = _current_cuda_device()
    compute_rank = int(config.spare_ranks[0])
    nbytes = group_nbytes[group[0].relative_index]
    raw_by_col: dict[int, torch.Tensor] = {}
    for slot in group:
        if slot.is_virtual_zero:
            continue
        assert slot.train_rank is not None
        src_rank = int(slot.train_rank)
        if rank == src_rank:
            payload = _payload_store_slot(
                local_slot_payload[src_rank],
                nbytes,
                device=device,
                zero_slot=zero_send_slot,
            )
            if compute_rank == rank:
                raw_by_col[slot.data_group_id] = payload
            else:
                _send_tensor(payload, compute_rank, process_group)
                bytes_sent += nbytes
        elif rank == compute_rank:
            raw_by_col[slot.data_group_id] = _recv_tensor(nbytes, src_rank, device=device, process_group=process_group)

    if rank == compute_rank:
        active_cols = [col for col in range(config.k) if col in raw_by_col]
        if not active_cols:
            raise RuntimeError("cannot compute parity for a reduction group with no real input columns")
        inputs = [raw_by_col[col] for col in active_cols]
        coeff = [[int(row[col]) for col in active_cols] for row in E[config.k : config.k + config.m]]
        parities = codec_cuda.apply_matrix_cuda(inputs, coeff)
        for parity_id, parity in enumerate(parities):
            owner = int(config.train_ranks[config.k + parity_id])
            chunk_id = _chunk_id(group[0].relative_index, config.k + parity_id)
            if owner == rank:
                local_chunks[chunk_id] = parity
            else:
                _send_tensor(parity, owner, process_group)
                bytes_sent += nbytes
    else:
        for parity_id in range(config.m):
            owner = int(config.train_ranks[config.k + parity_id])
            chunk_id = _chunk_id(group[0].relative_index, config.k + parity_id)
            if rank == owner:
                if receive_slot is None:
                    raise RuntimeError("strict RACER store requires a preallocated slot4/slot3 parity receive buffer")
                received = _recv_tensor_into(receive_slot, nbytes, compute_rank, process_group=process_group)
                if row_sink is None:
                    local_chunks[chunk_id] = received
                else:
                    row_sink(chunk_id, received)
    return bytes_sent


def _store_spare_compute_parity_cuda(
    *,
    rank: int,
    config: RacerConfig,
    layout: ElasticLayout,
    E: Sequence[Sequence[int]],
    group_nbytes: dict[int, int],
    local_slot_payload: dict[int, torch.Tensor],
    local_chunks: dict[str, torch.Tensor],
    process_group: Any | None = None,
) -> int:
    bytes_sent = 0
    for group in layout.reduction_groups:
        bytes_sent += _store_spare_compute_parity_cuda_for_group(
            rank=rank,
            config=config,
            group=group,
            E=E,
            group_nbytes=group_nbytes,
            local_slot_payload=local_slot_payload,
            local_chunks=local_chunks,
            receive_slot=None,
            zero_send_slot=None,
            row_sink=None,
            process_group=process_group,
        )
    return bytes_sent


def distributed_store(
    *,
    config: RacerConfig,
    local_packet: torch.Tensor | None,
    tag: str,
    process_group: Any | None = None,
    chunk_storage: Any | None = None,
    packet_sizes_by_rank: Mapping[int, int] | None = None,
) -> DistributedStoreResult:
    """Store one local train-rank packet per process.

    By default this uses the default torch.distributed process group.  Passing a
    low-level NCCL ``process_group`` allows callers such as Megatron adapters to
    keep their training world unchanged while adding a spare-GPU worker in a
    separate communication domain.
    """

    store_start = time.perf_counter()
    if chunk_storage is None:
        raise RuntimeError(
            "distributed_store requires daemon-owned checkpoint storage; "
            "in-process/local-chunk fallback is disabled"
        )
    _require_nccl(process_group)
    rank = _rank(process_group)
    if rank in config.spare_ranks and local_packet is not None:
        raise ValueError("spare ranks must pass local_packet=None")
    if rank in config.train_ranks and local_packet is None:
        raise ValueError(f"train rank {rank} must pass its local checkpoint packet")

    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    E = cauchy.generate_systematic_matrix(config.k, config.m, config.w, optimize=config.optimize_cauchy)
    local_payload = _cuda_payload(local_packet)
    setup_ms = (time.perf_counter() - store_start) * 1000.0

    sizing_start = time.perf_counter()
    if packet_sizes_by_rank is None:
        packet_sizes = _packet_sizes_by_rank(config, local_payload, process_group)
    else:
        packet_sizes = {int(rank): int(nbytes) for rank, nbytes in dict(packet_sizes_by_rank).items()}
        missing = [int(rank) for rank in config.train_ranks if int(rank) not in packet_sizes]
        if missing:
            raise ValueError(f"packet_sizes_by_rank is missing train ranks {missing}")
    group_sizes = _group_nbytes(layout, packet_sizes)
    plan = routing.make_planner(config).plan(layout, E, max(group_sizes.values(), default=0))
    local_slot_payload = {rank: local_payload} if rank in config.train_ranks and local_payload is not None else {}
    local_chunks: dict[str, torch.Tensor] = {}
    sizing_ms = (time.perf_counter() - sizing_start) * 1000.0

    manifest_start = time.perf_counter()
    manifest = _build_storage_manifest(
        tag=tag,
        config=config,
        layout=layout,
        matrix=E,
        plan=plan,
        group_nbytes=group_sizes,
        packet_sizes=packet_sizes,
    )
    manifest_ms = (time.perf_counter() - manifest_start) * 1000.0
    state = DistributedStoreResult(
        tag=tag,
        config=config,
        layout=layout,
        matrix=E,
        plan=plan,
        local_chunks=local_chunks,
        packet_nbytes_by_rank=packet_sizes,
        manifest=manifest,
        profile={
            "setup_ms": setup_ms,
            "sizing_ms": sizing_ms,
            "data_rows_ms": 0.0,
            "parity_ms": 0.0,
            "final_barrier_ms": 0.0,
            "manifest_ms": manifest_ms,
            "storage_ms": 0.0,
            "group_transfer_barrier_ms": 0.0,
            "group_storage_barrier_ms": 0.0,
            "data_rows_bytes_sent": 0,
            "parity_bytes_sent": 0,
            "local_storage_nbytes": 0,
            "local_storage_chunk_count": 0,
        },
    )
    storage_start = time.perf_counter()
    storage_profile = _begin_storage_checkpoint(
        chunk_storage=chunk_storage,
        state=state,
        rank=rank,
        process_group=process_group,
    )
    data_rows_ms = 0.0
    parity_ms = 0.0
    data_bytes_sent = 0
    parity_bytes_sent = 0
    group_transfer_barrier_ms = 0.0
    group_storage_barrier_ms = 0.0
    stored_nbytes = 0
    stored_count = 0
    released_nbytes = 0
    released_count = 0
    max_group_nbytes = max(group_sizes.values(), default=0)
    recv_slot4 = (
        torch.empty(int(max_group_nbytes), dtype=torch.uint8, device=_current_cuda_device())
        if rank in config.train_ranks and int(max_group_nbytes) > 0
        else None
    )
    try:
        local_source_group = layout.locate_rank(rank).relative_index if rank in config.train_ranks else None
    except KeyError:
        local_source_group = None
    source_group_completed = rank not in config.train_ranks

    def receive_slot_for(group: Sequence[Any], *, after_source_send: bool) -> torch.Tensor | None:
        # Keep local_payload immutable until all daemon writes for this store call are complete.
        # Reusing slot3 for receives can overwrite a source buffer whose direct-IPC EGM copy
        # is still visible to the daemon/runtime on GB200. recv_slot4 is already preallocated
        # for the largest group and is flushed before reuse across groups.
        return recv_slot4

    def flush_ready(chunks: Mapping[str, torch.Tensor]) -> None:
        nonlocal stored_nbytes, stored_count, released_nbytes, released_count
        if not chunks:
            return
        released_nbytes += sum(int(chunk.numel()) for chunk in chunks.values())
        released_count += len(chunks)
        group_stored_nbytes, group_stored_count, group_storage_profile = _put_storage_chunks(
            chunk_storage=chunk_storage,
            state=state,
            chunks=chunks,
            rank=rank,
        )
        stored_nbytes += int(group_stored_nbytes)
        stored_count += int(group_stored_count)
        _accumulate_profile(storage_profile, group_storage_profile)

    for group in layout.reduction_groups:
        group_chunks: dict[str, torch.Tensor] = {}
        group_index = int(group[0].relative_index)
        # Keep the valid-width view here.  The data/parity send helpers widen
        # it with the group's reusable zero/receive slot when required.
        group_local_slot_payload = local_slot_payload
        data_rows_start = time.perf_counter()
        data_bytes_sent += _store_data_rows_for_group(
            rank=rank,
            config=config,
            group=group,
            group_nbytes=group_sizes,
            local_slot_payload=group_local_slot_payload,
            local_chunks=group_chunks,
            receive_slot=receive_slot_for(group, after_source_send=False),
            zero_send_slot=recv_slot4,
            row_sink=None,
            process_group=process_group,
        )
        data_rows_ms += (time.perf_counter() - data_rows_start) * 1000.0

        parity_start = time.perf_counter()
        parity_bytes_sent += _store_spare_compute_parity_cuda_for_group(
            rank=rank,
            config=config,
            group=group,
            E=E,
            group_nbytes=group_sizes,
            local_slot_payload=group_local_slot_payload,
            local_chunks=group_chunks,
            receive_slot=receive_slot_for(
                group,
                after_source_send=local_source_group is not None
                and int(group[0].relative_index) == int(local_source_group),
            ),
            zero_send_slot=recv_slot4,
            row_sink=None,
            process_group=process_group,
        )
        parity_ms += (time.perf_counter() - parity_start) * 1000.0
        transfer_barrier_start = time.perf_counter()
        _barrier(process_group)
        group_transfer_barrier_ms += (time.perf_counter() - transfer_barrier_start) * 1000.0
        flush_ready(group_chunks)
        group_chunks.clear()
        storage_barrier_start = time.perf_counter()
        _barrier(process_group)
        group_storage_barrier_ms += (time.perf_counter() - storage_barrier_start) * 1000.0
        if local_source_group is not None and int(group[0].relative_index) == int(local_source_group):
            source_group_completed = True

    commit_profile = _commit_storage_checkpoint(
        chunk_storage=chunk_storage,
        state=state,
        rank=rank,
        process_group=process_group,
    )
    _accumulate_profile(storage_profile, commit_profile)
    if state.profile is not None:
        state.profile["storage_ms"] = (time.perf_counter() - storage_start) * 1000.0
        state.profile["data_rows_ms"] = data_rows_ms
        state.profile["parity_ms"] = parity_ms
        state.profile["group_transfer_barrier_ms"] = group_transfer_barrier_ms
        state.profile["group_storage_barrier_ms"] = group_storage_barrier_ms
        state.profile["data_rows_bytes_sent"] = int(data_bytes_sent)
        state.profile["parity_bytes_sent"] = int(parity_bytes_sent)
        state.profile["local_storage_nbytes"] = int(stored_nbytes)
        state.profile["local_storage_chunk_count"] = int(stored_count)
        state.profile.update(storage_profile)
    if state.profile is not None:
        state.profile["total_ms"] = (time.perf_counter() - store_start) * 1000.0
    state.local_chunks.clear()
    state.storage_backed = True
    if state.profile is not None:
        state.profile["local_chunks_released_nbytes"] = int(released_nbytes)
        state.profile["local_chunks_released_count"] = int(released_count)
    return state


def prepare_distributed_store_many(
    *,
    config: RacerConfig,
    tags: Sequence[str],
    local_packets: Sequence[torch.Tensor | None],
    process_group: Any | None = None,
    chunk_storage: Any | None = None,
    packet_sizes_by_tag: Mapping[str, Mapping[int, int]] | None = None,
) -> DistributedStoreBatchHandle:
    """Prepare a batch on the caller thread and enqueue strict CUDA-IPC writes.

    Each tag keeps its own manifest, erasure-code layout, and daemon-owned chunks.
    This phase owns every RACER NCCL/EC launch and every runtime-process-group
    barrier. It enqueues daemon CUDA-IPC PUTs but deliberately does not wait or
    commit them, so callers may move the returned handle to a background thread.

    ``packet_sizes_by_tag`` follows the same semantics as
    :func:`distributed_store`'s ``packet_sizes_by_rank`` argument, keyed by tag.
    Train-rank packet tensors must have disjoint storage because every packet
    remains live in the handle until :func:`finalize_distributed_store_many`
    completes its local CSD waits.
    """

    batch_start = time.perf_counter()
    normalized_tags = [str(tag) for tag in tags]
    packets = list(local_packets)
    if len(normalized_tags) != len(packets):
        raise ValueError(
            "prepare_distributed_store_many requires one local packet per tag; "
            f"tags={len(normalized_tags)}, local_packets={len(packets)}"
        )
    if len(set(normalized_tags)) != len(normalized_tags):
        raise ValueError("prepare_distributed_store_many requires unique tags")
    if not normalized_tags:
        return DistributedStoreBatchHandle(
            states=[],
            chunk_storage=chunk_storage,
            rank=0,
            pending_puts_by_tag=[],
            retained_tensors_by_tag=[],
            storage_profiles=[],
            execute_ms_by_tag=[],
            batch_start=batch_start,
            prepare_total_ms=(time.perf_counter() - batch_start) * 1000.0,
        )
    if chunk_storage is None:
        raise RuntimeError(
            "prepare_distributed_store_many requires daemon-owned checkpoint storage; "
            "in-process/local-chunk fallback is disabled"
        )

    _require_nccl(process_group)
    rank = _rank(process_group)
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    E = cauchy.generate_systematic_matrix(config.k, config.m, config.w, optimize=config.optimize_cauchy)
    supplied_sizes = {
        str(tag): {int(owner): int(nbytes) for owner, nbytes in dict(sizes).items()}
        for tag, sizes in dict(packet_sizes_by_tag or {}).items()
    }

    prepared: list[tuple[DistributedStoreResult, torch.Tensor | None, dict[int, int]]] = []
    storage_profiles: list[dict[str, float]] = []
    payload_ranges: list[tuple[torch.device, int, int, str]] = []
    for tag, local_packet in zip(normalized_tags, packets):
        store_start = time.perf_counter()
        if rank in config.spare_ranks and local_packet is not None:
            raise ValueError("spare ranks must pass local_packet=None")
        if rank in config.train_ranks and local_packet is None:
            raise ValueError(f"train rank {rank} must pass its local checkpoint packet")

        local_payload = _cuda_payload(local_packet)
        if local_payload is not None and int(local_payload.numel()) > 0:
            payload_start = int(local_payload.data_ptr())
            payload_end = payload_start + int(local_payload.numel()) * int(local_payload.element_size())
            for prior_device, prior_start, prior_end, prior_tag in payload_ranges:
                if local_payload.device == prior_device and payload_start < prior_end and prior_start < payload_end:
                    raise ValueError(
                        "prepare_distributed_store_many train-rank packets must not alias; "
                        f"tags {prior_tag!r} and {tag!r} overlap the same staging storage. "
                        "Use a distinct CUDA staging slot for every batched tag."
                    )
            payload_ranges.append(
                (local_payload.device, payload_start, payload_end, tag)
            )
        setup_ms = (time.perf_counter() - store_start) * 1000.0
        sizing_start = time.perf_counter()
        if tag not in supplied_sizes:
            packet_sizes = _packet_sizes_by_rank(config, local_payload, process_group)
        else:
            packet_sizes = dict(supplied_sizes[tag])
            missing = [int(owner) for owner in config.train_ranks if int(owner) not in packet_sizes]
            if missing:
                raise ValueError(f"packet_sizes_by_tag[{tag!r}] is missing train ranks {missing}")
        group_sizes = _group_nbytes(layout, packet_sizes)
        plan = routing.make_planner(config).plan(layout, E, max(group_sizes.values(), default=0))
        sizing_ms = (time.perf_counter() - sizing_start) * 1000.0

        manifest_start = time.perf_counter()
        manifest = _build_storage_manifest(
            tag=tag,
            config=config,
            layout=layout,
            matrix=E,
            plan=plan,
            group_nbytes=group_sizes,
            packet_sizes=packet_sizes,
        )
        manifest_ms = (time.perf_counter() - manifest_start) * 1000.0
        state = DistributedStoreResult(
            tag=tag,
            config=config,
            layout=layout,
            matrix=E,
            plan=plan,
            local_chunks={},
            packet_nbytes_by_rank=packet_sizes,
            manifest=manifest,
            profile={
                "setup_ms": setup_ms,
                "sizing_ms": sizing_ms,
                "data_rows_ms": 0.0,
                "parity_ms": 0.0,
                "final_barrier_ms": 0.0,
                "manifest_ms": manifest_ms,
                "storage_ms": 0.0,
                "group_transfer_barrier_ms": 0.0,
                "group_storage_barrier_ms": 0.0,
                "data_rows_bytes_sent": 0,
                "parity_bytes_sent": 0,
                "local_storage_nbytes": 0,
                "local_storage_chunk_count": 0,
            },
        )
        prepared.append((state, local_payload, group_sizes))
        storage_profiles.append(
            {
                "storage_begin_ms": 0.0,
                "storage_begin_barrier_ms": 0.0,
                "storage_enqueue_barrier_ms": 0.0,
                "storage_wait_ms": 0.0,
                "storage_commit_pre_barrier_ms": 0.0,
                "storage_commit_ms": 0.0,
                "storage_commit_post_barrier_ms": 0.0,
                "storage_commit_retry_count": 0.0,
            }
        )

    pending_puts_by_tag: list[list[DistributedStorePendingPut]] = [[] for _ in prepared]
    retained_tensors_by_tag: list[list[torch.Tensor]] = [
        ([local_payload] if local_payload is not None else [])
        for _state, local_payload, _group_sizes in prepared
    ]

    # Begin every child tag first. A single barrier then makes all manifests
    # visible before any rank starts sending data for the batch.
    for index, (state, _local_payload, _group_sizes) in enumerate(prepared):
        coordinator = int(state.config.train_ranks[0])
        if _per_node_csd_enabled():
            local_owner_ranks = _local_csd_owner_ranks(rank)
            should_begin = rank == _local_csd_coordinator_rank(rank)
            expected_chunks = _expected_storage_chunk_count_for_ranks(state.manifest, local_owner_ranks)
        else:
            should_begin = rank == coordinator
            expected_chunks = _expected_storage_chunk_count(state.manifest)
        if should_begin:
            begin_start = time.perf_counter()
            chunk_storage.begin(
                state.tag,
                manifest_base=state.manifest or {},
                expected_chunks=expected_chunks,
            )
            storage_profiles[index]["storage_begin_ms"] = (time.perf_counter() - begin_start) * 1000.0
    begin_barrier_start = time.perf_counter()
    _barrier(process_group)
    storage_profiles[-1]["storage_begin_barrier_ms"] = (
        time.perf_counter() - begin_barrier_start
    ) * 1000.0

    execute_ms_by_tag: list[float] = []
    for state, local_payload, group_sizes in prepared:
        execute_start = time.perf_counter()
        data_rows_ms = 0.0
        parity_ms = 0.0
        data_bytes_sent = 0
        parity_bytes_sent = 0
        group_transfer_barrier_ms = 0.0
        stored_nbytes = 0
        stored_count = 0
        state_index = len(execute_ms_by_tag)
        storage_profile = storage_profiles[state_index]

        for group in layout.reduction_groups:
            group_chunks: dict[str, torch.Tensor] = {}
            group_index = int(group[0].relative_index)
            group_nbytes = int(group_sizes[group_index])
            recv_slot4 = (
                torch.empty(group_nbytes, dtype=torch.uint8, device=_current_cuda_device())
                if rank in config.train_ranks and group_nbytes > 0
                else None
            )
            if recv_slot4 is not None:
                retained_tensors_by_tag[state_index].append(recv_slot4)
            # Delay tail padding until a send/placement actually consumes the
            # local packet so it can reuse this group's retained recv slot.
            group_local_slot_payload = (
                {rank: local_payload}
                if rank in config.train_ranks and local_payload is not None
                else {}
            )

            data_rows_start = time.perf_counter()
            data_bytes_sent += _store_data_rows_for_group(
                rank=rank,
                config=config,
                group=group,
                group_nbytes=group_sizes,
                local_slot_payload=group_local_slot_payload,
                local_chunks=group_chunks,
                receive_slot=recv_slot4,
                zero_send_slot=recv_slot4,
                row_sink=None,
                process_group=process_group,
            )
            data_rows_ms += (time.perf_counter() - data_rows_start) * 1000.0

            parity_start = time.perf_counter()
            parity_bytes_sent += _store_spare_compute_parity_cuda_for_group(
                rank=rank,
                config=config,
                group=group,
                E=E,
                group_nbytes=group_sizes,
                local_slot_payload=group_local_slot_payload,
                local_chunks=group_chunks,
                receive_slot=recv_slot4,
                zero_send_slot=recv_slot4,
                row_sink=None,
                process_group=process_group,
            )
            parity_ms += (time.perf_counter() - parity_start) * 1000.0

            transfer_barrier_start = time.perf_counter()
            _stream_barrier(process_group)
            group_transfer_barrier_ms += (
                time.perf_counter() - transfer_barrier_start
            ) * 1000.0

            group_stored_nbytes, group_stored_count, group_storage_profile, group_pending = (
                _enqueue_storage_chunks(
                    chunk_storage=chunk_storage,
                    state=state,
                    chunks=group_chunks,
                    rank=rank,
                )
            )
            pending_puts_by_tag[state_index].extend(group_pending)
            retained_tensors_by_tag[state_index].extend(put.tensor for put in group_pending)
            stored_nbytes += int(group_stored_nbytes)
            stored_count += int(group_stored_count)
            _accumulate_profile(storage_profile, group_storage_profile)
            # Every receive slot and every tensor exported through CUDA IPC stays
            # reachable from the handle until finalize waits the matching op.
            group_chunks.clear()

        execute_ms_by_tag.append((time.perf_counter() - execute_start) * 1000.0)
        assert state.profile is not None
        state.profile["data_rows_ms"] = data_rows_ms
        state.profile["parity_ms"] = parity_ms
        state.profile["group_transfer_barrier_ms"] = group_transfer_barrier_ms
        state.profile["group_storage_barrier_ms"] = 0.0
        state.profile["data_rows_bytes_sent"] = int(data_bytes_sent)
        state.profile["parity_bytes_sent"] = int(parity_bytes_sent)
        state.profile["local_storage_nbytes"] = int(stored_nbytes)
        state.profile["local_storage_chunk_count"] = int(stored_count)
        state.profile["local_chunks_retained_nbytes"] = int(stored_nbytes)
        state.profile["local_chunks_retained_count"] = int(stored_count)
        state.local_chunks.clear()

    return DistributedStoreBatchHandle(
        states=[state for state, _local_payload, _group_sizes in prepared],
        chunk_storage=chunk_storage,
        rank=rank,
        pending_puts_by_tag=pending_puts_by_tag,
        retained_tensors_by_tag=retained_tensors_by_tag,
        storage_profiles=storage_profiles,
        execute_ms_by_tag=execute_ms_by_tag,
        batch_start=batch_start,
        prepare_total_ms=(time.perf_counter() - batch_start) * 1000.0,
    )


_INCOMPLETE_CSD_COMMIT = re.compile(
    r"cannot commit: sealed_chunks=(?P<sealed>[0-9]+), expected_chunks=(?P<expected>[0-9]+)"
)


def _commit_storage_checkpoint_after_local_waits(
    *,
    chunk_storage: Any,
    state: DistributedStoreResult,
    rank: int,
) -> dict[str, float]:
    """Commit without any process-group operation, retrying only incomplete seals."""

    profile: dict[str, float] = {
        "storage_manifest_put_ms": 0.0,
        "storage_commit_pre_barrier_ms": 0.0,
        "storage_commit_ms": 0.0,
        "storage_commit_post_barrier_ms": 0.0,
        "storage_commit_retry_count": 0.0,
        "storage_commit_wait_ms": 0.0,
    }
    if _per_node_csd_enabled():
        should_commit = rank == _local_csd_coordinator_rank(rank)
    else:
        should_commit = rank == int(state.config.train_ranks[0])
    if not should_commit:
        return profile

    manifest_start = time.perf_counter()
    chunk_storage.put_manifest(state.tag, state.manifest or {})
    profile["storage_manifest_put_ms"] = (time.perf_counter() - manifest_start) * 1000.0

    timeout_s = max(
        0.001,
        float(os.environ.get("RACER_CSD_COMMIT_WAIT_TIMEOUT_SECONDS", "300")),
    )
    retry_interval_s = max(
        0.001,
        min(
            1.0,
            float(os.environ.get("RACER_CSD_COMMIT_RETRY_INTERVAL_MS", "10")) / 1000.0,
        ),
    )
    deadline = time.monotonic() + timeout_s
    commit_start = time.perf_counter()
    retries = 0
    retry_wait_ms = 0.0
    while True:
        try:
            chunk_storage.commit(state.tag)
            break
        except RuntimeError as exc:
            match = _INCOMPLETE_CSD_COMMIT.search(str(exc))
            if match is None:
                raise
            sealed = int(match.group("sealed"))
            expected = int(match.group("expected"))
            if sealed >= expected:
                raise
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise TimeoutError(
                    "timed out waiting for local CSD chunks to seal before commit; "
                    f"tag={state.tag!r}, sealed_chunks={sealed}, "
                    f"expected_chunks={expected}, timeout_s={timeout_s}"
                ) from exc
            retries += 1
            sleep_start = time.perf_counter()
            time.sleep(min(retry_interval_s, remaining_s))
            retry_wait_ms += (time.perf_counter() - sleep_start) * 1000.0
    profile["storage_commit_ms"] = (time.perf_counter() - commit_start) * 1000.0
    profile["storage_commit_retry_count"] = float(retries)
    profile["storage_commit_wait_ms"] = retry_wait_ms
    return profile


def finalize_distributed_store_many(
    handle: DistributedStoreBatchHandle,
) -> list[DistributedStoreResult]:
    """Wait and commit a prepared batch without touching torch.distributed/NCCL."""

    if handle.finalized:
        return handle.states
    if handle.finalize_started:
        raise RuntimeError("RACER distributed store batch finalization is already in progress")
    handle.finalize_started = True
    finalize_start = time.perf_counter()
    if not handle.states:
        handle.finalized = True
        return []

    for index, state in enumerate(handle.states):
        storage_profile = handle.storage_profiles[index]
        wait_profile = _wait_storage_puts(
            chunk_storage=handle.chunk_storage,
            pending=handle.pending_puts_by_tag[index],
        )
        _accumulate_profile(storage_profile, wait_profile)
        commit_profile = _commit_storage_checkpoint_after_local_waits(
            chunk_storage=handle.chunk_storage,
            state=state,
            rank=handle.rank,
        )
        _accumulate_profile(storage_profile, commit_profile)

        assert state.profile is not None
        state.profile.update(storage_profile)
        state.profile["storage_ms"] = (
            float(handle.execute_ms_by_tag[index])
            + float(storage_profile["storage_begin_ms"])
            + float(storage_profile["storage_begin_barrier_ms"])
            + float(storage_profile["storage_enqueue_barrier_ms"])
            + float(storage_profile["storage_wait_ms"])
            + float(storage_profile["storage_commit_ms"])
        )
        state.profile["total_ms"] = (
            float(state.profile["setup_ms"])
            + float(state.profile["sizing_ms"])
            + float(state.profile["manifest_ms"])
            + float(state.profile["storage_ms"])
        )
        state.profile["local_chunks_released_nbytes"] = int(
            state.profile["local_storage_nbytes"]
        )
        state.profile["local_chunks_released_count"] = int(
            state.profile["local_storage_chunk_count"]
        )
        state.storage_backed = True
        handle.pending_puts_by_tag[index].clear()
        handle.retained_tensors_by_tag[index].clear()

    finalize_total_ms = (time.perf_counter() - finalize_start) * 1000.0
    batch_total_ms = (time.perf_counter() - handle.batch_start) * 1000.0
    for index, state in enumerate(handle.states):
        assert state.profile is not None
        # Shared values live on one result so numeric per-tag aggregation remains exact.
        state.profile["store_batch_prepare_ms"] = handle.prepare_total_ms if index == 0 else 0.0
        state.profile["store_batch_finalize_ms"] = finalize_total_ms if index == 0 else 0.0
        state.profile["store_batch_total_ms"] = batch_total_ms if index == 0 else 0.0
        state.profile["store_batch_tag_count"] = len(handle.states) if index == 0 else 0
    handle.finalized = True
    return handle.states


def distributed_store_many(
    *,
    config: RacerConfig,
    tags: Sequence[str],
    local_packets: Sequence[torch.Tensor | None],
    process_group: Any | None = None,
    chunk_storage: Any | None = None,
    packet_sizes_by_tag: Mapping[str, Mapping[int, int]] | None = None,
) -> list[DistributedStoreResult]:
    """Synchronous compatibility wrapper for prepare/finalize batch storage."""

    handle = prepare_distributed_store_many(
        config=config,
        tags=tags,
        local_packets=local_packets,
        process_group=process_group,
        chunk_storage=chunk_storage,
        packet_sizes_by_tag=packet_sizes_by_tag,
    )
    return finalize_distributed_store_many(handle)


def distributed_state_from_storage(
    *,
    config: RacerConfig,
    tag: str,
    chunk_storage: Any,
    process_group: Any | None = None,
) -> DistributedStoreResult:
    """Rebuild a distributed store state from committed daemon-owned chunks."""

    load_start = time.perf_counter()
    _require_nccl(process_group)
    rank = _rank(process_group)
    device = _current_cuda_device()
    manifest_start = time.perf_counter()
    manifest = chunk_storage.get_manifest(tag)
    validate_committed_daemon_manifest(manifest, tag=tag)
    manifest_ms = (time.perf_counter() - manifest_start) * 1000.0
    setup_start = time.perf_counter()
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    matrix = [list(row) for row in manifest.get("E") or cauchy.generate_systematic_matrix(
        config.k,
        config.m,
        config.w,
        optimize=config.optimize_cauchy,
    )]
    packet_sizes = {
        int(rank_id): int(nbytes)
        for rank_id, nbytes in dict(manifest.get("packet_nbytes_by_rank", {})).items()
    }
    if not packet_sizes:
        packet_sizes = {int(rank_id): 0 for rank_id in config.train_ranks}
    group_sizes = {
        int(group_id): int(nbytes)
        for group_id, nbytes in dict(manifest.get("group_nbytes", {})).items()
    }
    if not group_sizes:
        group_sizes = _group_nbytes(layout, packet_sizes)
    plan = routing.make_planner(config).plan(layout, matrix, max(group_sizes.values(), default=0))
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    local_chunks: dict[str, torch.Tensor] = {}
    chunk_by_id = {str(chunk["chunk_id"]): dict(chunk) for chunk in manifest.get("chunks", [])}
    zero_chunks = _zero_data_chunk_ids(manifest)
    zero_cache: dict[tuple[int | None, int], torch.Tensor] = {}
    read_futures: list[tuple[str, str, torch.Tensor]] = []
    storage_profile: dict[str, float] = {}
    read_checksum_verified = 0
    read_checksum_mismatch = 0
    read_checksum_ms = 0.0
    read_nbytes = 0
    read_count = 0
    use_cuda_ipc_read = (
        hasattr(chunk_storage, "read_into_cuda_tensor")
        and hasattr(chunk_storage, "wait")
        and bool(dict(chunk_storage.capabilities()).get("supports_cuda_ipc", False))
        if hasattr(chunk_storage, "capabilities")
        else False
    )
    read_enqueue_ms = 0.0
    read_wait_ms = 0.0
    for chunk in manifest.get("chunks", []):
        if int(chunk.get("owner_rank", -1)) != rank:
            continue
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in zero_chunks:
            local_chunks[chunk_id] = _zero_buffer(int(chunk.get("num_bytes", 0)), device=device, zero_cache=zero_cache)
        elif use_cuda_ipc_read and device.type == "cuda":
            nbytes = int(chunk.get("nbytes", chunk.get("num_bytes", 0)) or 0)
            dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
            enqueue_start = time.perf_counter()
            op_id = chunk_storage.read_into_cuda_tensor(tag, chunk_id, dst)
            read_enqueue_ms += (time.perf_counter() - enqueue_start) * 1000.0
            read_nbytes += nbytes
            read_count += 1
            wait_start = time.perf_counter()
            result = chunk_storage.wait(op_id)
            read_wait_ms += (time.perf_counter() - wait_start) * 1000.0
            if isinstance(result, dict):
                profile = result.get("profile")
                if isinstance(profile, dict):
                    for key, value in profile.items():
                        if isinstance(value, (int, float)):
                            storage_profile[f"csd_get_{key}"] = storage_profile.get(f"csd_get_{key}", 0.0) + float(value)
            verified, mismatch, checksum_ms = _debug_storage_read_checksum(
                tag=tag,
                rank=rank,
                chunk_id=chunk_id,
                chunk=chunk,
                tensor=dst,
            )
            read_checksum_verified += verified
            read_checksum_mismatch += mismatch
            read_checksum_ms += checksum_ms
            local_chunks[chunk_id] = dst
        else:
            raise RuntimeError(
                "RACER distributed load requires daemon storage with CUDA IPC async read support; "
                "CPU/socket fallback get is disabled"
            )
    barrier_start = time.perf_counter()
    _barrier(process_group)
    final_barrier_ms = (time.perf_counter() - barrier_start) * 1000.0
    profile: dict[str, Any] = {
        "load_manifest_ms": manifest_ms,
        "load_setup_ms": setup_ms,
        "load_read_enqueue_ms": read_enqueue_ms,
        "load_read_wait_ms": read_wait_ms,
        "load_final_barrier_ms": final_barrier_ms,
        "load_total_ms": (time.perf_counter() - load_start) * 1000.0,
        "load_read_nbytes": int(read_nbytes),
        "load_read_chunk_count": int(read_count),
        "load_cuda_ipc_read": bool(use_cuda_ipc_read),
        "storage_read_checksum_verified_count": int(read_checksum_verified),
        "storage_read_checksum_mismatch_count": int(read_checksum_mismatch),
        "storage_read_checksum_ms": float(read_checksum_ms),
    }
    profile.update(storage_profile)
    return DistributedStoreResult(
        tag=str(tag),
        config=config,
        layout=layout,
        matrix=matrix,
        plan=plan,
        local_chunks=local_chunks,
        packet_nbytes_by_rank=packet_sizes,
        manifest=manifest,
        profile=profile,
    )


def distributed_load_from_storage(
    *,
    config: RacerConfig,
    tag: str,
    chunk_storage: Any,
    failed_train_ranks: Sequence[int],
    requested_train_ranks: Sequence[int] | None = None,
    replacement_mapping: dict[int, int] | None = None,
    process_group: Any | None = None,
) -> DistributedLoadResult:
    state = distributed_state_from_storage(
        config=config,
        tag=tag,
        chunk_storage=chunk_storage,
        process_group=process_group,
    )
    result = distributed_load(
        state=state,
        failed_train_ranks=failed_train_ranks,
        requested_train_ranks=requested_train_ranks,
        replacement_mapping=replacement_mapping,
        process_group=process_group,
    )
    profile = dict(state.profile or {})
    profile.update(dict(result.profile or {}))
    result.profile = profile
    return result


def distributed_load_local_payload_from_storage(
    *,
    config: RacerConfig,
    tag: str,
    chunk_storage: Any,
    requested_train_rank: int,
    device: torch.device | None = None,
) -> DistributedLoadResult:
    """Load one requested train-rank payload directly from committed storage.

    This is the no-failure restart fast path.  It still uses the committed RACER
    manifest and stored data/parity chunk layout, but it does not create the
    spare-rank NCCL runtime because no decode or repair is needed.
    """

    load_start = time.perf_counter()
    if device is None:
        device = _current_cuda_device()
    requested = int(requested_train_rank)
    manifest_start = time.perf_counter()
    manifest = chunk_storage.get_manifest(tag)
    validate_committed_daemon_manifest(manifest, tag=tag)
    manifest_ms = (time.perf_counter() - manifest_start) * 1000.0
    setup_start = time.perf_counter()
    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    packet_sizes = {
        int(rank_id): int(nbytes)
        for rank_id, nbytes in dict(manifest.get("packet_nbytes_by_rank", {})).items()
    }
    if not packet_sizes:
        packet_sizes = {int(rank_id): 0 for rank_id in config.train_ranks}
    slot = layout.locate_rank(requested)
    chunk_id = _chunk_id(slot.relative_index, slot.data_group_id)
    chunk_by_id = {str(chunk["chunk_id"]): dict(chunk) for chunk in manifest.get("chunks", [])}
    chunk = chunk_by_id.get(chunk_id)
    if chunk is None:
        raise KeyError(f"RACER storage manifest does not contain chunk {chunk_id!r} for rank {requested}")
    valid_nbytes = int(packet_sizes.get(requested, 0))
    storage_nbytes = int(chunk.get("nbytes", chunk.get("num_bytes", valid_nbytes)) or valid_nbytes)
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    zero_chunks = _zero_data_chunk_ids(manifest)
    storage_profile: dict[str, float] = {}
    read_checksum_verified = 0
    read_checksum_mismatch = 0
    read_checksum_ms = 0.0
    read_enqueue_ms = 0.0
    read_wait_ms = 0.0
    if chunk_id in zero_chunks or valid_nbytes <= 0:
        payload = _zero_buffer(valid_nbytes, device=device)
    else:
        use_cuda_ipc_read = (
            hasattr(chunk_storage, "read_into_cuda_tensor")
            and hasattr(chunk_storage, "wait")
            and bool(dict(chunk_storage.capabilities()).get("supports_cuda_ipc", False))
            if hasattr(chunk_storage, "capabilities")
            else False
        )
        if use_cuda_ipc_read and device.type == "cuda":
            enqueue_start = time.perf_counter()
            dst = torch.empty(storage_nbytes, dtype=torch.uint8, device=device)
            op_id = chunk_storage.read_into_cuda_tensor(tag, chunk_id, dst)
            read_enqueue_ms = (time.perf_counter() - enqueue_start) * 1000.0
            wait_start = time.perf_counter()
            result = chunk_storage.wait(op_id)
            read_wait_ms = (time.perf_counter() - wait_start) * 1000.0
            if isinstance(result, dict):
                profile = result.get("profile")
                if isinstance(profile, dict):
                    for key, value in profile.items():
                        if isinstance(value, (int, float)):
                            storage_profile[f"csd_get_{key}"] = storage_profile.get(f"csd_get_{key}", 0.0) + float(value)
            verified, mismatch, checksum_ms = _debug_storage_read_checksum(
                tag=tag,
                rank=requested,
                chunk_id=chunk_id,
                chunk=chunk,
                tensor=dst,
            )
            read_checksum_verified += verified
            read_checksum_mismatch += mismatch
            read_checksum_ms += checksum_ms
        else:
            raise RuntimeError(
                "RACER distributed fetch requires daemon storage with CUDA IPC async read support; "
                "CPU/socket fallback get is disabled"
            )
        payload = dst.narrow(0, 0, valid_nbytes)
        if not payload.is_contiguous():
            payload = payload.contiguous()
    profile: dict[str, Any] = {
        "load_manifest_ms": manifest_ms,
        "load_setup_ms": setup_ms,
        "load_read_enqueue_ms": read_enqueue_ms,
        "load_read_wait_ms": read_wait_ms,
        "load_final_barrier_ms": 0.0,
        "load_total_ms": (time.perf_counter() - load_start) * 1000.0,
        "load_read_nbytes": int(storage_nbytes if valid_nbytes > 0 else 0),
        "load_read_chunk_count": int(0 if chunk_id in zero_chunks or valid_nbytes <= 0 else 1),
        "load_cuda_ipc_read": bool(storage_profile or read_wait_ms > 0.0),
        "storage_read_checksum_verified_count": int(read_checksum_verified),
        "storage_read_checksum_mismatch_count": int(read_checksum_mismatch),
        "storage_read_checksum_ms": float(read_checksum_ms),
        "load_direct_storage": True,
        "load_route_pre_barrier_ms": 0.0,
        "load_route_requested_ms": 0.0,
        "load_route_decode_ms": 0.0,
        "load_route_final_barrier_ms": 0.0,
        "load_route_total_ms": 0.0,
        "load_decode_request_count": 0.0,
    }
    profile.update(storage_profile)
    return DistributedLoadResult(recovered={requested: payload}, decode_rank=-1, profile=profile)


def _choose_decode_rank(config: RacerConfig, failed_train_ranks: Sequence[int]) -> int:
    return int(config.spare_ranks[0])


def _failed_owner_rows(config: RacerConfig, failed_train_ranks: Sequence[int]) -> set[int]:
    failed = {int(rank) for rank in failed_train_ranks}
    return {
        row
        for row, owner in enumerate(config.train_ranks)
        if int(owner) in failed
    }


def _load_requested_payloads(
    *,
    state: DistributedStoreResult,
    rank: int,
    requested_train_ranks: Sequence[int],
    failed_train_ranks: set[int],
    failed_owner_rows: set[int],
    process_group: Any | None = None,
) -> dict[int, torch.Tensor]:
    config = state.config
    layout = state.layout
    device = _current_cuda_device()
    loaded: dict[int, torch.Tensor] = {}
    for requested_rank in [int(value) for value in requested_train_ranks]:
        if requested_rank in failed_train_ranks:
            continue
        slot = layout.locate_rank(requested_rank)
        if slot.data_group_id in failed_owner_rows:
            continue
        owner = int(config.train_ranks[slot.data_group_id])
        chunk_id = _chunk_id(slot.relative_index, slot.data_group_id)
        nbytes = int(state.packet_nbytes_by_rank[requested_rank])
        if rank == owner:
            payload = state.local_chunks[chunk_id].narrow(0, 0, nbytes)
            if not payload.is_contiguous():
                payload = payload.contiguous()
            if owner == requested_rank:
                loaded[requested_rank] = payload
            else:
                _send_tensor(payload, requested_rank, process_group)
        elif rank == requested_rank:
            loaded[requested_rank] = _recv_tensor(nbytes, owner, device=device, process_group=process_group)
    return loaded


def distributed_load(
    *,
    state: DistributedStoreResult,
    failed_train_ranks: Sequence[int],
    requested_train_ranks: Sequence[int] | None = None,
    replacement_mapping: dict[int, int] | None = None,
    process_group: Any | None = None,
) -> DistributedLoadResult:
    """Load or recover train-rank packets with distributed P2P survivor fetches.

    ``failed_train_ranks`` requests erasure-code recovery on the spare rank.
    ``requested_train_ranks`` requests ordinary in-memory loads for nonfailed
    ranks, routed from the data-row owner back to the original train rank.
    """

    _require_nccl(process_group)
    rank = _rank(process_group)
    config = state.config
    layout = state.layout
    device = _current_cuda_device()
    failed = [int(value) for value in failed_train_ranks]
    failed_set = set(failed)
    failed_owner_rows = _failed_owner_rows(config, failed)
    replacement = {int(k): int(v) for k, v in dict(replacement_mapping or {}).items()}
    decode_rank = _choose_decode_rank(config, failed)
    recovered: dict[int, torch.Tensor] = {}
    load_start = time.perf_counter()

    def output_rank_for(requested_rank: int) -> int:
        requested_rank = int(requested_rank)
        if requested_rank in replacement:
            return int(replacement[requested_rank])
        if requested_rank in failed_set:
            if not config.spare_ranks:
                raise RuntimeError(
                    f"failed train rank {requested_rank} has no replacement mapping and no spare rank fallback"
                )
            return int(config.spare_ranks[0])
        return requested_rank

    if requested_train_ranks is None:
        requested = list(failed)
    else:
        requested = [int(value) for value in requested_train_ranks]

    pre_barrier_start = time.perf_counter()
    _barrier(process_group)
    pre_barrier_ms = (time.perf_counter() - pre_barrier_start) * 1000.0
    requested_start = time.perf_counter()
    recovered.update(
        _load_requested_payloads(
            state=state,
            rank=rank,
            requested_train_ranks=requested,
            failed_train_ranks=failed_set,
            failed_owner_rows=failed_owner_rows,
            process_group=process_group,
        )
    )
    requested_ms = (time.perf_counter() - requested_start) * 1000.0

    decode_requests = []
    for requested_rank in requested:
        slot = layout.locate_rank(requested_rank)
        if requested_rank in failed_set or slot.data_group_id in failed_owner_rows:
            decode_requests.append((int(requested_rank), slot))

    # Store pads every codeword row to the aligned reduction-group width.  P2P
    # send/recv counts during decode must use that exact width; using only the
    # largest valid packet silently mismatches NCCL counts on the tail chunk.
    # Recompute for old manifests, then prefer the recorded storage width.
    stored_group_nbytes = _group_nbytes(layout, state.packet_nbytes_by_rank)
    stored_group_nbytes.update(
        {
            int(group_id): int(value)
            for group_id, value in dict((state.manifest or {}).get("group_nbytes", {})).items()
        }
    )
    decode_start = time.perf_counter()
    for requested_rank, slot in decode_requests:
        survivors = [row for row in range(len(config.train_ranks)) if row not in failed_owner_rows]
        if len(survivors) < config.k:
            raise RuntimeError(f"not enough survivor rows to decode: have {len(survivors)}, need {config.k}")
        chosen_rows = survivors[: config.k]
        valid_group_nbytes = max(
            state.packet_nbytes_by_rank[int(s.train_rank)]
            for s in layout.reduction_groups[slot.relative_index]
            if s.train_rank is not None
        )
        nbytes = int(stored_group_nbytes.get(int(slot.relative_index), valid_group_nbytes))
        if nbytes < valid_group_nbytes:
            raise RuntimeError(
                "RACER manifest group_nbytes is smaller than a valid packet: "
                f"group={slot.relative_index}, stored={nbytes}, valid={valid_group_nbytes}"
            )
        survivor_chunks: list[torch.Tensor] = []
        for row in chosen_rows:
            owner = int(config.train_ranks[row])
            chunk_id = _chunk_id(slot.relative_index, row)
            if rank == owner:
                chunk = state.local_chunks[chunk_id]
                if int(chunk.numel()) < nbytes:
                    raise RuntimeError(
                        "RACER survivor chunk is smaller than manifest group_nbytes: "
                        f"chunk_id={chunk_id}, chunk_nbytes={int(chunk.numel())}, "
                        f"group_nbytes={nbytes}"
                    )
                chunk = chunk.narrow(0, 0, nbytes)
                if owner == decode_rank:
                    survivor_chunks.append(chunk)
                else:
                    _send_tensor(chunk, decode_rank, process_group)
            elif rank == decode_rank:
                survivor_chunks.append(_recv_tensor(nbytes, owner, device=device, process_group=process_group))

        if rank == decode_rank:
            decoded = codec_cuda.decode_blocks(survivor_chunks, chosen_rows, state.matrix)
            valid = state.packet_nbytes_by_rank[requested_rank]
            payload = decoded[slot.data_group_id][:valid].contiguous()
            output_rank = output_rank_for(requested_rank)
            if output_rank == decode_rank:
                recovered[requested_rank] = payload
            else:
                _send_tensor(payload, output_rank, process_group)
        elif rank == output_rank_for(requested_rank):
            recovered[requested_rank] = _recv_tensor(
                state.packet_nbytes_by_rank[requested_rank],
                decode_rank,
                device=device,
                process_group=process_group,
            )

    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    final_barrier_start = time.perf_counter()
    _barrier(process_group)
    final_barrier_ms = (time.perf_counter() - final_barrier_start) * 1000.0
    return DistributedLoadResult(
        recovered=recovered,
        decode_rank=decode_rank,
        profile={
            "load_route_pre_barrier_ms": pre_barrier_ms,
            "load_route_requested_ms": requested_ms,
            "load_route_decode_ms": decode_ms,
            "load_route_final_barrier_ms": final_barrier_ms,
            "load_route_total_ms": (time.perf_counter() - load_start) * 1000.0,
            "load_decode_request_count": float(len(decode_requests)),
        },
    )


def plan_as_dict(plan: RoutingPlan) -> dict:
    return asdict(plan)
