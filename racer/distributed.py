"""Distributed RACER executor for torchrun prototypes.

This module implements real cross-process paths for the 5-GPU examples:

* ``training_local`` uses CPU byte buffers, a Gloo process group, and Jerasure
  so it models ECCHECK placement: each train rank computes GF products locally
  on CPU, sends encoded contributions to reduction owners, and owners
  XOR-reduce.
* ``spare_compute`` uses CUDA byte buffers, the default NCCL process group, and
  RACER CUDA GF kernels so raw packets go to the spare GPU for compute.

Neither path serializes Python state_dict objects; both operate on raw tensor
byte buffers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch
import torch.distributed as dist

from . import cauchy, codec_cpu, codec_cuda, gf256, jerasure, routing
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
    used_jerasure: bool
    comm_backend: str
    compute_backend: str

    @property
    def routing_cost(self):
        return self.plan.cost


@dataclass
class DistributedLoadResult:
    recovered: dict[int, torch.Tensor]
    decode_rank: int


_CPU_GROUP = None


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


def _rank() -> int:
    _require_dist()
    return int(dist.get_rank())


def _world_size() -> int:
    _require_dist()
    return int(dist.get_world_size())


def _current_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NCCL distributed RACER paths")
    return torch.device("cuda", torch.cuda.current_device())


def _default_backend() -> str:
    _require_dist()
    return str(dist.get_backend())


def _cpu_group():
    global _CPU_GROUP
    if _default_backend() == "gloo":
        return None
    if _CPU_GROUP is None:
        _CPU_GROUP = dist.new_group(backend="gloo")
    return _CPU_GROUP


def _comm_group(backend: str):
    if backend == "gloo":
        return _cpu_group()
    if backend == "nccl":
        return None
    raise ValueError(f"unsupported distributed comm backend: {backend}")


def _barrier(backend: str) -> None:
    dist.barrier(group=_comm_group(backend))


def _chunk_id(stripe_index: int, row: int) -> str:
    return f"stripe_{stripe_index:06d}_row_{row:03d}"


def _cpu_payload(local_packet: torch.Tensor | None) -> torch.Tensor | None:
    if local_packet is None:
        return None
    if local_packet.dtype != torch.uint8:
        raise TypeError("distributed RACER packets must be torch.uint8")
    # This is a raw byte-buffer copy/view, not object serialization. GPU packets
    # move to host only because the Jerasure route models CPU computation.
    return local_packet.detach().contiguous().view(-1).cpu()


def _cuda_payload(local_packet: torch.Tensor | None) -> torch.Tensor | None:
    if local_packet is None:
        return None
    if local_packet.dtype != torch.uint8:
        raise TypeError("distributed RACER packets must be torch.uint8")
    if local_packet.device.type != "cuda":
        raise ValueError("NCCL spare_compute requires CUDA local packets")
    return local_packet.detach().contiguous().view(-1)


def _all_gather_int(value: int, *, backend: str) -> list[int]:
    device = _current_cuda_device() if backend == "nccl" else torch.device("cpu")
    tensor = torch.tensor([int(value)], dtype=torch.long, device=device)
    gathered = [torch.zeros_like(tensor) for _ in range(_world_size())]
    dist.all_gather(gathered, tensor, group=_comm_group(backend))
    return [int(item.item()) for item in gathered]


def _pad(payload: torch.Tensor | None, nbytes: int, *, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = payload.device if payload is not None else torch.device("cpu")
    out = torch.zeros(int(nbytes), dtype=torch.uint8, device=device)
    if payload is not None:
        flat = payload.contiguous().view(-1)
        out[: min(int(flat.numel()), int(nbytes))].copy_(flat[:nbytes])
    return out


def _xor_inplace(dst: torch.Tensor, src: torch.Tensor) -> None:
    dst.bitwise_xor_(src)


def _mul_cpu(src: torch.Tensor, coeff: int, use_jerasure: bool) -> torch.Tensor:
    coeff = int(coeff) & 0xFF
    if coeff == 0:
        return torch.zeros_like(src)
    if coeff == 1:
        return src.clone()
    if use_jerasure and jerasure.available():
        return jerasure.region_multiply(src, coeff)
    return gf256.mul_tensor(src, coeff)


def _send_tensor(tensor: torch.Tensor, dst: int, *, backend: str) -> None:
    if backend == "nccl" and tensor.device.type != "cuda":
        raise ValueError("NCCL send requires a CUDA tensor")
    if backend == "gloo" and tensor.device.type != "cpu":
        raise ValueError("Gloo CPU send requires a CPU tensor")
    dist.send(tensor.contiguous(), dst=int(dst), group=_comm_group(backend))


def _recv_tensor(nbytes: int, src: int, *, backend: str, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = _current_cuda_device() if backend == "nccl" else torch.device("cpu")
    out = torch.empty(int(nbytes), dtype=torch.uint8, device=device)
    dist.recv(out, src=int(src), group=_comm_group(backend))
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
    *,
    backend: str,
) -> dict[int, int]:
    rank = _rank()
    local_nbytes = int(local_payload.numel()) if rank in config.train_ranks and local_payload is not None else 0
    gathered = _all_gather_int(local_nbytes, backend=backend)
    return {int(train_rank): int(gathered[int(train_rank)]) for train_rank in config.train_ranks}


def _store_data_rows(
    *,
    rank: int,
    config: RacerConfig,
    layout: ElasticLayout,
    group_nbytes: dict[int, int],
    local_slot_payload: dict[int, torch.Tensor],
    local_chunks: dict[str, torch.Tensor],
    backend: str,
) -> int:
    bytes_sent = 0
    device = _current_cuda_device() if backend == "nccl" else torch.device("cpu")
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
                    _send_tensor(payload, owner, backend=backend)
                    bytes_sent += nbytes
            elif rank == owner:
                local_chunks[chunk_id] = _recv_tensor(nbytes, src_rank, backend=backend, device=device)
    return bytes_sent


def _store_training_local_parity(
    *,
    rank: int,
    config: RacerConfig,
    layout: ElasticLayout,
    E: Sequence[Sequence[int]],
    group_nbytes: dict[int, int],
    local_slot_payload: dict[int, torch.Tensor],
    local_chunks: dict[str, torch.Tensor],
    use_jerasure: bool,
) -> int:
    bytes_sent = 0
    for group in layout.reduction_groups:
        nbytes = group_nbytes[group[0].relative_index]
        for parity_id in range(config.m):
            owner = int(config.train_ranks[config.k + parity_id])
            chunk_id = _chunk_id(group[0].relative_index, config.k + parity_id)
            if rank == owner:
                local_chunks[chunk_id] = torch.zeros(nbytes, dtype=torch.uint8)
            for slot in group:
                if slot.is_virtual_zero:
                    continue
                assert slot.train_rank is not None
                src_rank = int(slot.train_rank)
                coeff = int(E[config.k + parity_id][slot.data_group_id]) & 0xFF
                if rank == src_rank:
                    payload = _pad(local_slot_payload[src_rank], nbytes, device=torch.device("cpu"))
                    contrib = _mul_cpu(payload, coeff, use_jerasure)
                    if owner == rank:
                        _xor_inplace(local_chunks[chunk_id], contrib)
                    else:
                        _send_tensor(contrib, owner, backend="gloo")
                        bytes_sent += nbytes
                elif rank == owner:
                    contrib = _recv_tensor(nbytes, src_rank, backend="gloo", device=torch.device("cpu"))
                    _xor_inplace(local_chunks[chunk_id], contrib)
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
) -> int:
    bytes_sent = 0
    device = _current_cuda_device()
    compute_rank = int(config.spare_ranks[0] if config.spare_ranks else config.train_ranks[0])
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
                    _send_tensor(payload, compute_rank, backend="nccl")
                    bytes_sent += nbytes
            elif rank == compute_rank:
                raw_by_col[slot.data_group_id] = _recv_tensor(nbytes, src_rank, backend="nccl", device=device)

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
                    _send_tensor(parity, owner, backend="nccl")
                    bytes_sent += nbytes
        else:
            for parity_id in range(config.m):
                owner = int(config.train_ranks[config.k + parity_id])
                chunk_id = _chunk_id(group[0].relative_index, config.k + parity_id)
                if rank == owner:
                    local_chunks[chunk_id] = _recv_tensor(nbytes, compute_rank, backend="nccl", device=device)
    torch.cuda.synchronize(device)
    return bytes_sent


def _route_backends(plan: RoutingPlan) -> tuple[str, str, bool]:
    if plan.strategy == "spare_compute":
        return "nccl", "cuda", False
    return "gloo", "jerasure_cpu", True


def distributed_store(
    *,
    config: RacerConfig,
    local_packet: torch.Tensor | None,
    tag: str,
    use_jerasure: bool = True,
) -> DistributedStoreResult:
    """Store one local train-rank packet per torchrun process.

    Every process in the default process group must call this function. Train
    ranks pass their local checkpoint packet; spare ranks pass None.
    """

    _require_dist()
    rank = _rank()
    if rank in config.spare_ranks and local_packet is not None:
        raise ValueError("spare ranks must pass local_packet=None")
    if rank in config.train_ranks and local_packet is None:
        raise ValueError(f"train rank {rank} must pass its local checkpoint packet")

    layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
    E = cauchy.generate_systematic_matrix(config.k, config.m, config.w, optimize=config.optimize_cauchy)
    sizing_backend = "nccl" if config.routing_strategy in {"spare_compute", "hybrid"} else "gloo"
    if sizing_backend == "nccl" and _default_backend() != "nccl":
        raise RuntimeError("spare_compute/hybrid sizing requires the default process group backend to be NCCL")

    sizing_payload = _cuda_payload(local_packet) if sizing_backend == "nccl" else _cpu_payload(local_packet)
    packet_sizes = _packet_sizes_by_rank(config, sizing_payload, backend=sizing_backend)
    group_sizes = _group_nbytes(layout, packet_sizes)
    plan = routing.make_planner(config).plan(layout, E, max(group_sizes.values(), default=0))
    comm_backend, compute_backend, route_uses_jerasure = _route_backends(plan)
    if comm_backend == "nccl" and _default_backend() != "nccl":
        raise RuntimeError("spare_compute requires the default process group backend to be NCCL")

    local_payload = _cuda_payload(local_packet) if comm_backend == "nccl" else _cpu_payload(local_packet)
    local_slot_payload = {rank: local_payload} if rank in config.train_ranks and local_payload is not None else {}
    local_chunks: dict[str, torch.Tensor] = {}

    _barrier(comm_backend)
    _store_data_rows(
        rank=rank,
        config=config,
        layout=layout,
        group_nbytes=group_sizes,
        local_slot_payload=local_slot_payload,
        local_chunks=local_chunks,
        backend=comm_backend,
    )

    if plan.strategy == "spare_compute":
        _store_spare_compute_parity_cuda(
            rank=rank,
            config=config,
            layout=layout,
            E=E,
            group_nbytes=group_sizes,
            local_slot_payload=local_slot_payload,
            local_chunks=local_chunks,
        )
    else:
        _store_training_local_parity(
            rank=rank,
            config=config,
            layout=layout,
            E=E,
            group_nbytes=group_sizes,
            local_slot_payload=local_slot_payload,
            local_chunks=local_chunks,
            use_jerasure=use_jerasure,
        )
    _barrier(comm_backend)

    used_jerasure = bool(route_uses_jerasure and use_jerasure and jerasure.available())
    return DistributedStoreResult(
        tag=tag,
        config=config,
        layout=layout,
        matrix=E,
        plan=plan,
        local_chunks=local_chunks,
        packet_nbytes_by_rank=packet_sizes,
        used_jerasure=used_jerasure,
        comm_backend=comm_backend,
        compute_backend=compute_backend,
    )


def _failed_rows(config: RacerConfig, failed_train_ranks: Iterable[int]) -> set[int]:
    row_by_rank = {int(rank): row for row, rank in enumerate(config.train_ranks)}
    return {row_by_rank[int(rank)] for rank in failed_train_ranks}


def _choose_decode_rank(config: RacerConfig, failed_train_ranks: Sequence[int]) -> int:
    if config.spare_ranks:
        return int(config.spare_ranks[0])
    failed = {int(rank) for rank in failed_train_ranks}
    for rank in config.train_ranks:
        if int(rank) not in failed:
            return int(rank)
    raise RuntimeError("no live rank is available to decode")


def distributed_load(
    *,
    state: DistributedStoreResult,
    failed_train_ranks: Sequence[int],
) -> DistributedLoadResult:
    """Recover failed train-rank packets with distributed P2P survivor fetches."""

    _require_dist()
    rank = _rank()
    config = state.config
    layout = state.layout
    backend = state.comm_backend
    device = _current_cuda_device() if backend == "nccl" else torch.device("cpu")
    failed = [int(value) for value in failed_train_ranks]
    decode_rank = _choose_decode_rank(config, failed)
    failed_rows = _failed_rows(config, failed)
    survivors = [row for row in range(len(config.train_ranks)) if row not in failed_rows]
    if len(survivors) < config.k:
        raise RuntimeError(f"not enough survivor rows to decode: have {len(survivors)}, need {config.k}")
    chosen_rows = survivors[: config.k]
    recovered: dict[int, torch.Tensor] = {}

    _barrier(backend)
    for failed_rank in failed:
        slot = layout.locate_rank(failed_rank)
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
                    _send_tensor(chunk, decode_rank, backend=backend)
            elif rank == decode_rank:
                survivor_chunks.append(_recv_tensor(nbytes, owner, backend=backend, device=device))

        if rank == decode_rank:
            if backend == "nccl":
                decoded = codec_cuda.decode_blocks(survivor_chunks, chosen_rows, state.matrix)
            else:
                decoded = codec_cpu.decode_cpu(survivor_chunks, chosen_rows, state.matrix)
            valid = state.packet_nbytes_by_rank[failed_rank]
            payload = decoded[slot.data_group_id][:valid].contiguous()
            if failed_rank == decode_rank:
                recovered[failed_rank] = payload
            else:
                _send_tensor(payload, failed_rank, backend=backend)
        elif rank == failed_rank:
            recovered[failed_rank] = _recv_tensor(
                state.packet_nbytes_by_rank[failed_rank],
                decode_rank,
                backend=backend,
                device=device,
            )

    _barrier(backend)
    return DistributedLoadResult(recovered=recovered, decode_rank=decode_rank)


def plan_as_dict(plan: RoutingPlan) -> dict:
    return asdict(plan)
