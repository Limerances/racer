"""Distributed RACER executor for torchrun spare-GPU prototypes.

This module implements the NCCL spare-compute path only: train ranks send raw
CUDA byte buffers to the spare GPU, the spare GPU computes parity with RACER
CUDA GF kernels, and parity rows are returned to train-rank chunk owners.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist

from . import cauchy, codec_cuda, routing
from .config import RacerConfig
from .layout import ElasticLayout
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

    @property
    def routing_cost(self):
        return self.plan.cost


@dataclass
class DistributedLoadResult:
    recovered: dict[int, torch.Tensor]
    decode_rank: int


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


def _cuda_payload(local_packet: torch.Tensor | None) -> torch.Tensor | None:
    if local_packet is None:
        return None
    if local_packet.dtype != torch.uint8:
        raise TypeError("distributed RACER packets must be torch.uint8")
    if local_packet.device.type != "cuda":
        raise ValueError("distributed RACER spare_compute requires CUDA local packets")
    return local_packet.detach().contiguous().view(-1)


def _all_gather_int(value: int, process_group: Any | None = None) -> list[int]:
    device = _current_cuda_device()
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    gathered = [torch.zeros_like(tensor) for _ in range(_world_size(process_group))]
    if process_group is None:
        dist.all_gather(gathered, tensor)
    else:
        process_group.allgather(gathered, tensor).wait()
    return [int(item.item()) for item in gathered]


def _pad(payload: torch.Tensor | None, nbytes: int, *, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = payload.device if payload is not None else _current_cuda_device()
    if device.type != "cuda":
        raise ValueError("distributed RACER buffers must stay on CUDA devices")
    out = torch.zeros(int(nbytes), dtype=torch.uint8, device=device)
    if payload is not None:
        flat = payload.contiguous().view(-1)
        out[: min(int(flat.numel()), int(nbytes))].copy_(flat[:nbytes])
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
    device = _current_cuda_device()
    for group in layout.reduction_groups:
        nbytes = group_nbytes[group[0].relative_index]
        for slot in group:
            owner = int(config.train_ranks[slot.data_group_id])
            chunk_id = _chunk_id(slot.relative_index, slot.data_group_id)
            if slot.is_virtual_zero:
                if rank == owner:
                    local_chunks[chunk_id] = torch.zeros(nbytes, dtype=torch.uint8, device=device)
                continue
            assert slot.train_rank is not None
            src_rank = int(slot.train_rank)
            if rank == src_rank:
                payload = _pad(local_slot_payload[src_rank], nbytes, device=device)
                if owner == rank:
                    local_chunks[chunk_id] = payload.clone()
                else:
                    _send_tensor(payload, owner, process_group)
                    bytes_sent += nbytes
            elif rank == owner:
                local_chunks[chunk_id] = _recv_tensor(nbytes, src_rank, device=device, process_group=process_group)
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
    device = _current_cuda_device()
    compute_rank = int(config.spare_ranks[0])
    for group in layout.reduction_groups:
        nbytes = group_nbytes[group[0].relative_index]
        raw_by_col: dict[int, torch.Tensor] = {}
        for slot in group:
            if slot.is_virtual_zero:
                raw_by_col[slot.data_group_id] = torch.zeros(nbytes, dtype=torch.uint8, device=device)
                continue
            assert slot.train_rank is not None
            src_rank = int(slot.train_rank)
            if rank == src_rank:
                payload = _pad(local_slot_payload[src_rank], nbytes, device=device)
                if compute_rank == rank:
                    raw_by_col[slot.data_group_id] = payload.clone()
                else:
                    _send_tensor(payload, compute_rank, process_group)
                    bytes_sent += nbytes
            elif rank == compute_rank:
                raw_by_col[slot.data_group_id] = _recv_tensor(nbytes, src_rank, device=device, process_group=process_group)

        if rank == compute_rank:
            inputs = [raw_by_col[col] for col in range(config.k)]
            coeff = [list(row) for row in E[config.k : config.k + config.m]]
            parities = codec_cuda.apply_matrix_cuda(inputs, coeff)
            for parity_id, parity in enumerate(parities):
                owner = int(config.train_ranks[config.k + parity_id])
                chunk_id = _chunk_id(group[0].relative_index, config.k + parity_id)
                if owner == rank:
                    local_chunks[chunk_id] = parity.clone()
                else:
                    _send_tensor(parity, owner, process_group)
                    bytes_sent += nbytes
        else:
            for parity_id in range(config.m):
                owner = int(config.train_ranks[config.k + parity_id])
                chunk_id = _chunk_id(group[0].relative_index, config.k + parity_id)
                if rank == owner:
                    local_chunks[chunk_id] = _recv_tensor(nbytes, compute_rank, device=device, process_group=process_group)
    torch.cuda.synchronize(device)
    return bytes_sent


def distributed_store(
    *,
    config: RacerConfig,
    local_packet: torch.Tensor | None,
    tag: str,
    process_group: Any | None = None,
) -> DistributedStoreResult:
    """Store one local train-rank packet per process.

    By default this uses the default torch.distributed process group.  Passing a
    low-level NCCL ``process_group`` allows callers such as Megatron adapters to
    keep their training world unchanged while adding a spare-GPU worker in a
    separate communication domain.
    """

    _require_nccl(process_group)
    rank = _rank(process_group)
    if rank in config.spare_ranks and local_packet is not None:
        raise ValueError("spare ranks must pass local_packet=None")
    if rank in config.train_ranks and local_packet is None:
        raise ValueError(f"train rank {rank} must pass its local checkpoint packet")

    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    E = cauchy.generate_systematic_matrix(config.k, config.m, config.w, optimize=config.optimize_cauchy)
    local_payload = _cuda_payload(local_packet)
    packet_sizes = _packet_sizes_by_rank(config, local_payload, process_group)
    group_sizes = _group_nbytes(layout, packet_sizes)
    plan = routing.make_planner(config).plan(layout, E, max(group_sizes.values(), default=0))
    local_slot_payload = {rank: local_payload} if rank in config.train_ranks and local_payload is not None else {}
    local_chunks: dict[str, torch.Tensor] = {}

    _barrier(process_group)
    _store_data_rows(
        rank=rank,
        config=config,
        layout=layout,
        group_nbytes=group_sizes,
        local_slot_payload=local_slot_payload,
        local_chunks=local_chunks,
        process_group=process_group,
    )
    _store_spare_compute_parity_cuda(
        rank=rank,
        config=config,
        layout=layout,
        E=E,
        group_nbytes=group_sizes,
        local_slot_payload=local_slot_payload,
        local_chunks=local_chunks,
        process_group=process_group,
    )
    _barrier(process_group)

    return DistributedStoreResult(
        tag=tag,
        config=config,
        layout=layout,
        matrix=E,
        plan=plan,
        local_chunks=local_chunks,
        packet_nbytes_by_rank=packet_sizes,
    )


def _choose_decode_rank(config: RacerConfig, failed_train_ranks: Sequence[int]) -> int:
    return int(config.spare_ranks[0])


def _load_requested_payloads(
    *,
    state: DistributedStoreResult,
    rank: int,
    requested_train_ranks: Sequence[int],
    failed_train_ranks: set[int],
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
        owner = int(config.train_ranks[slot.data_group_id])
        chunk_id = _chunk_id(slot.relative_index, slot.data_group_id)
        nbytes = int(state.packet_nbytes_by_rank[requested_rank])
        if rank == owner:
            payload = state.local_chunks[chunk_id][:nbytes].contiguous()
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
    decode_rank = _choose_decode_rank(config, failed)
    recovered: dict[int, torch.Tensor] = {}

    if requested_train_ranks is None:
        requested = list(failed)
    else:
        requested = [int(value) for value in requested_train_ranks]

    _barrier(process_group)
    recovered.update(
        _load_requested_payloads(
            state=state,
            rank=rank,
            requested_train_ranks=requested,
            failed_train_ranks=failed_set,
            process_group=process_group,
        )
    )

    failed_cols_by_reduction_group: dict[int, set[int]] = {}
    for failed_rank in failed:
        slot = layout.locate_rank(failed_rank)
        failed_cols_by_reduction_group.setdefault(slot.relative_index, set()).add(slot.data_group_id)

    for failed_rank in failed:
        slot = layout.locate_rank(failed_rank)
        failed_cols = failed_cols_by_reduction_group[slot.relative_index]
        survivors = [row for row in range(len(config.train_ranks)) if row not in failed_cols]
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
                    survivor_chunks.append(chunk.clone())
                else:
                    _send_tensor(chunk, decode_rank, process_group)
            elif rank == decode_rank:
                survivor_chunks.append(_recv_tensor(nbytes, owner, device=device, process_group=process_group))

        if rank == decode_rank:
            decoded = codec_cuda.decode_blocks(survivor_chunks, chosen_rows, state.matrix)
            valid = state.packet_nbytes_by_rank[failed_rank]
            payload = decoded[slot.data_group_id][:valid].contiguous()
            if failed_rank == decode_rank:
                recovered[failed_rank] = payload
            else:
                _send_tensor(payload, failed_rank, process_group)
        elif rank == failed_rank:
            recovered[failed_rank] = _recv_tensor(
                state.packet_nbytes_by_rank[failed_rank],
                decode_rank,
                device=device,
                process_group=process_group,
            )

    _barrier(process_group)
    return DistributedLoadResult(recovered=recovered, decode_rank=decode_rank)


def plan_as_dict(plan: RoutingPlan) -> dict:
    return asdict(plan)
