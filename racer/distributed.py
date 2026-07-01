"""Distributed RACER executor for torchrun spare-GPU prototypes.

This module implements the NCCL spare-compute path only: train ranks send raw
CUDA byte buffers to the spare GPU, the spare GPU computes parity with RACER
CUDA GF kernels, and parity rows are returned to train-rank chunk owners.
"""

from __future__ import annotations

import os
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


def _put_storage_chunks(
    *,
    chunk_storage: Any,
    state: DistributedStoreResult,
    chunks: Mapping[str, torch.Tensor],
    rank: int,
) -> tuple[int, int, dict[str, float]]:
    zero_chunks = _zero_data_chunk_ids(state.manifest)
    storage_profile: dict[str, float] = {
        "storage_enqueue_ms": 0.0,
        "storage_wait_ms": 0.0,
    }
    stored_nbytes = 0
    stored_count = 0
    futures: list[tuple[str, torch.Tensor]] = []
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
            futures.append((op_id, chunk))
        else:
            raise RuntimeError(
                "RACER distributed store requires daemon storage with CUDA IPC async write support; "
                "CPU/socket fallback put is disabled"
            )
        stored_nbytes += int(chunk.numel())
        stored_count += 1
    storage_profile["storage_enqueue_ms"] = (time.perf_counter() - enqueue_start) * 1000.0
    wait_start = time.perf_counter()
    for op_id, _chunk in futures:
        result = chunk_storage.wait(op_id)
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
    """Return a store-time slot3 view, padding in-place when the payload view is short."""
    flat = _flat_uint8_view(payload)
    nbytes = int(nbytes)
    if flat.device != device:
        raise ValueError("store payload slot must stay on the current CUDA device")
    if int(flat.numel()) == nbytes:
        return flat
    if int(flat.numel()) > nbytes:
        return flat.narrow(0, 0, nbytes)
    if int(flat.numel()) == 0 and nbytes > 0:
        if zero_slot is None:
            raise RuntimeError(
                "strict RACER store requires a fixed zero-send slot for empty payload chunks; "
                f"required_nbytes={nbytes}"
            )
        if zero_slot.device != device or zero_slot.device.type != "cuda":
            raise ValueError("strict RACER zero-send slot must be on the current CUDA device")
        if int(zero_slot.numel()) < nbytes:
            raise RuntimeError(
                "strict RACER zero-send slot is too small; "
                f"slot_nbytes={int(zero_slot.numel())}, required_nbytes={nbytes}"
            )
        out = zero_slot.narrow(0, 0, nbytes)
        out.zero_()
        return out

    storage_nbytes = int(flat.untyped_storage().nbytes())
    element_size = int(flat.element_size())
    required_storage_nbytes = (int(flat.storage_offset()) + nbytes) * element_size
    if required_storage_nbytes > storage_nbytes:
        raise RuntimeError(
            "strict RACER store requires the payload chunk to come from a full-size device slot; "
            f"valid_nbytes={int(flat.numel())}, required_nbytes={nbytes}, "
            f"storage_nbytes={storage_nbytes}"
        )
    slot = torch.as_strided(flat, (nbytes,), (1,), storage_offset=int(flat.storage_offset()))
    slot.narrow(0, int(flat.numel()), nbytes - int(flat.numel())).zero_()
    return slot


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
    for group in layout.reduction_groups:
        size = 0
        for slot in group:
            if slot.train_rank is not None:
                size = max(size, packet_sizes[int(slot.train_rank)])
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

    def slot3_view(nbytes: int) -> torch.Tensor:
        if local_payload is None:
            raise RuntimeError("strict RACER store cannot reuse slot3 without a local payload slot")
        return _payload_store_slot(
            local_payload,
            int(nbytes),
            device=_current_cuda_device(),
            zero_slot=recv_slot4,
        )

    def receive_slot_for(group: Sequence[Any], *, after_source_send: bool) -> torch.Tensor | None:
        nonlocal source_group_completed
        nbytes = int(group_sizes[group[0].relative_index])
        if (
            local_payload is not None
            and int(local_payload.numel()) > 0
            and (source_group_completed or after_source_send)
        ):
            return slot3_view(nbytes)
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

    def store_received_row(chunk_id: str, tensor: torch.Tensor) -> None:
        flush_ready({str(chunk_id): tensor})

    for group in layout.reduction_groups:
        group_chunks: dict[str, torch.Tensor] = {}
        data_rows_start = time.perf_counter()
        data_bytes_sent += _store_data_rows_for_group(
            rank=rank,
            config=config,
            group=group,
            group_nbytes=group_sizes,
            local_slot_payload=local_slot_payload,
            local_chunks=group_chunks,
            receive_slot=receive_slot_for(group, after_source_send=False),
            zero_send_slot=recv_slot4,
            row_sink=store_received_row,
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
            local_slot_payload=local_slot_payload,
            local_chunks=group_chunks,
            receive_slot=receive_slot_for(
                group,
                after_source_send=local_source_group is not None
                and int(group[0].relative_index) == int(local_source_group),
            ),
            zero_send_slot=recv_slot4,
            row_sink=store_received_row,
            process_group=process_group,
        )
        parity_ms += (time.perf_counter() - parity_start) * 1000.0
        flush_ready(group_chunks)
        group_chunks.clear()
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
    zero_chunks = _zero_data_chunk_ids(manifest)
    zero_cache: dict[tuple[int | None, int], torch.Tensor] = {}
    read_futures: list[tuple[str, str, torch.Tensor]] = []
    storage_profile: dict[str, float] = {}
    read_nbytes = 0
    read_count = 0
    use_cuda_ipc_read = (
        hasattr(chunk_storage, "read_into_cuda_tensor")
        and hasattr(chunk_storage, "wait")
        and bool(dict(chunk_storage.capabilities()).get("supports_cuda_ipc", False))
        if hasattr(chunk_storage, "capabilities")
        else False
    )
    read_enqueue_start = time.perf_counter()
    for chunk in manifest.get("chunks", []):
        if int(chunk.get("owner_rank", -1)) != rank:
            continue
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in zero_chunks:
            local_chunks[chunk_id] = _zero_buffer(int(chunk.get("num_bytes", 0)), device=device, zero_cache=zero_cache)
        elif use_cuda_ipc_read and device.type == "cuda":
            nbytes = int(chunk.get("nbytes", chunk.get("num_bytes", 0)) or 0)
            dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
            op_id = chunk_storage.read_into_cuda_tensor(tag, chunk_id, dst)
            read_futures.append((op_id, chunk_id, dst))
            read_nbytes += nbytes
            read_count += 1
        else:
            raise RuntimeError(
                "RACER distributed load requires daemon storage with CUDA IPC async read support; "
                "CPU/socket fallback get is disabled"
            )
    read_enqueue_ms = (time.perf_counter() - read_enqueue_start) * 1000.0
    read_wait_start = time.perf_counter()
    for op_id, chunk_id, dst in read_futures:
        result = chunk_storage.wait(op_id)
        if isinstance(result, dict):
            profile = result.get("profile")
            if isinstance(profile, dict):
                for key, value in profile.items():
                    if isinstance(value, (int, float)):
                        storage_profile[f"csd_get_{key}"] = storage_profile.get(f"csd_get_{key}", 0.0) + float(value)
        local_chunks[chunk_id] = dst
    read_wait_ms = (time.perf_counter() - read_wait_start) * 1000.0
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

    decode_start = time.perf_counter()
    for requested_rank, slot in decode_requests:
        survivors = [row for row in range(len(config.train_ranks)) if row not in failed_owner_rows]
        if len(survivors) < config.k:
            raise RuntimeError(f"not enough survivor rows to decode: have {len(survivors)}, need {config.k}")
        chosen_rows = survivors[: config.k]
        nbytes = max(
            state.packet_nbytes_by_rank[int(s.train_rank)]
            for s in layout.reduction_groups[slot.relative_index]
            if s.train_rank is not None
        )
        survivor_chunks: list[torch.Tensor] = []
        for row in chosen_rows:
            owner = int(config.train_ranks[row])
            chunk_id = _chunk_id(slot.relative_index, row)
            if rank == owner:
                chunk = state.local_chunks[chunk_id]
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
