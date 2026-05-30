"""Routing planner for spare-GPU checkpoint coding.

The planner models where RACER sends raw packets, GF multiply work, XOR
reductions, and final parity/repair results. Parity owners remain train
ranks; spare ranks are compute accelerators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .config import RacerConfig
from .layout import ElasticLayout, ElasticSlot


def storage_device_for_row(config: RacerConfig, row: int) -> torch.device:
    return torch.device("cuda", int(config.train_ranks[int(row)]))


def compute_device(config: RacerConfig) -> torch.device:
    if not config.spare_ranks:
        raise ValueError("spare_compute requires at least one spare rank")
    return torch.device("cuda", int(config.spare_ranks[0]))


def output_device_for_rank(config: RacerConfig, rank: int, failed: bool) -> torch.device:
    if failed:
        return torch.device("cuda", int(config.spare_ranks[0]))
    return torch.device("cuda", int(rank))


@dataclass(frozen=True)
class TransferOp:
    src_rank: int | None
    dst_rank: int
    src_device: str
    dst_device: str
    slot_id: int
    data_group_id: int
    relative_index: int
    parity_id: int | None
    coeff: int
    num_bytes: int
    is_virtual_zero: bool
    description: str


@dataclass(frozen=True)
class ComputeOp:
    rank: int
    device: str
    op_type: str
    coeff: int
    input_slot: int | None
    output_slot: str
    num_bytes: int
    is_on_train_rank: bool
    is_on_spare_rank: bool


@dataclass(frozen=True)
class ReductionOp:
    reduction_group_id: int
    parity_id: int
    target_rank: int
    participants: tuple[int, ...]
    skipped_virtual_zero_slots: tuple[int, ...]
    num_inputs: int
    num_bytes: int


@dataclass(frozen=True)
class CostModel:
    total_bytes_sent: int = 0
    num_messages: int = 0
    compute_bytes_on_train_ranks: int = 0
    compute_bytes_on_accelerators: int = 0
    xor_bytes_on_train_ranks: int = 0
    xor_bytes_on_accelerators: int = 0
    max_train_rank_compute_bytes: int = 0
    max_spare_compute_bytes: int = 0
    skipped_virtual_zero_bytes: int = 0
    estimated_critical_path: int = 0


@dataclass(frozen=True)
class RoutingPlan:
    transfers: tuple[TransferOp, ...] = field(default_factory=tuple)
    computes: tuple[ComputeOp, ...] = field(default_factory=tuple)
    reductions: tuple[ReductionOp, ...] = field(default_factory=tuple)
    cost: CostModel = field(default_factory=CostModel)


class _CostAccumulator:
    def __init__(self) -> None:
        self.total_bytes_sent = 0
        self.num_messages = 0
        self.compute_train: dict[int, int] = {}
        self.compute_spare: dict[int, int] = {}
        self.xor_train: dict[int, int] = {}
        self.xor_spare: dict[int, int] = {}
        self.skipped_virtual_zero_bytes = 0

    def transfer(self, num_bytes: int) -> None:
        self.total_bytes_sent += int(num_bytes)
        self.num_messages += 1

    def compute(self, rank: int, num_bytes: int, is_spare: bool) -> None:
        target = self.compute_spare if is_spare else self.compute_train
        target[int(rank)] = target.get(int(rank), 0) + int(num_bytes)

    def xor(self, rank: int, num_bytes: int, is_spare: bool) -> None:
        target = self.xor_spare if is_spare else self.xor_train
        target[int(rank)] = target.get(int(rank), 0) + int(num_bytes)

    def skip_virtual_zero(self, num_bytes: int) -> None:
        self.skipped_virtual_zero_bytes += int(num_bytes)

    def build(self) -> CostModel:
        train_compute = sum(self.compute_train.values())
        spare_compute = sum(self.compute_spare.values())
        train_xor = sum(self.xor_train.values())
        spare_xor = sum(self.xor_spare.values())
        max_train = max(self.compute_train.values(), default=0) + max(self.xor_train.values(), default=0)
        max_spare = max(self.compute_spare.values(), default=0) + max(self.xor_spare.values(), default=0)
        return CostModel(
            total_bytes_sent=self.total_bytes_sent,
            num_messages=self.num_messages,
            compute_bytes_on_train_ranks=train_compute,
            compute_bytes_on_accelerators=spare_compute,
            xor_bytes_on_train_ranks=train_xor,
            xor_bytes_on_accelerators=spare_xor,
            max_train_rank_compute_bytes=max_train,
            max_spare_compute_bytes=max_spare,
            skipped_virtual_zero_bytes=self.skipped_virtual_zero_bytes,
            estimated_critical_path=max(self.total_bytes_sent, max_train, max_spare),
        )


class SpareComputePlanner:
    def __init__(self, config: RacerConfig) -> None:
        self.config = config

    def _rank_device(self, rank: int) -> str:
        return f"cuda:{int(rank)}"

    def _spare_target(self) -> int:
        return int(self.config.spare_ranks[0])

    def _parity_owner(self, parity_id: int) -> int:
        return int(self.config.train_ranks[self.config.k + int(parity_id)])

    def _data_owner(self, data_group_id: int) -> int:
        return int(self.config.train_ranks[int(data_group_id)])

    def _coeff(self, E: Sequence[Sequence[int]], parity_id: int, data_group_id: int) -> int:
        return int(E[self.config.k + int(parity_id)][int(data_group_id)]) & 0xFF

    def plan(self, layout: ElasticLayout, E: Sequence[Sequence[int]], chunk_nbytes: int) -> RoutingPlan:
        transfers: list[TransferOp] = []
        computes: list[ComputeOp] = []
        reductions: list[ReductionOp] = []
        cost = _CostAccumulator()
        compute_rank = self._spare_target()

        for group in layout.reduction_groups:
            for slot in group:
                if slot.is_virtual_zero:
                    cost.skip_virtual_zero(chunk_nbytes)
                    continue
                data_owner = self._data_owner(slot.data_group_id)
                if slot.train_rank != data_owner:
                    transfers.append(self._transfer(slot, slot.train_rank, data_owner, None, 1, chunk_nbytes, "raw data chunk to code owner"))
                    cost.transfer(chunk_nbytes)
                if int(slot.train_rank) != compute_rank:
                    transfers.append(self._transfer(slot, slot.train_rank, compute_rank, None, 1, chunk_nbytes, "raw packet to spare compute target"))
                    cost.transfer(chunk_nbytes)

            for parity_id in range(self.config.m):
                participants: list[int] = []
                skipped: list[int] = []
                for slot in group:
                    if slot.is_virtual_zero:
                        skipped.append(slot.slot_id)
                        continue
                    coeff = self._coeff(E, parity_id, slot.data_group_id)
                    computes.append(
                        ComputeOp(
                            rank=compute_rank,
                            device=self._rank_device(compute_rank),
                            op_type="mul_xor",
                            coeff=coeff,
                            input_slot=slot.slot_id,
                            output_slot=f"rg{slot.relative_index}:p{parity_id}:reduction",
                            num_bytes=chunk_nbytes,
                            is_on_train_rank=False,
                            is_on_spare_rank=True,
                        )
                    )
                    cost.compute(compute_rank, chunk_nbytes, is_spare=True)
                    cost.xor(compute_rank, chunk_nbytes, is_spare=True)
                    participants.append(int(slot.train_rank))

                reductions.append(
                    ReductionOp(
                        reduction_group_id=group[0].relative_index,
                        parity_id=parity_id,
                        target_rank=compute_rank,
                        participants=tuple(participants),
                        skipped_virtual_zero_slots=tuple(skipped),
                        num_inputs=len(participants),
                        num_bytes=chunk_nbytes,
                    )
                )
                owner = self._parity_owner(parity_id)
                if compute_rank != owner:
                    transfers.append(self._transfer(group[0], compute_rank, owner, parity_id, 1, chunk_nbytes, "parity result to train-rank chunk owner"))
                    cost.transfer(chunk_nbytes)

        return RoutingPlan(tuple(transfers), tuple(computes), tuple(reductions), cost.build())

    def _transfer(
        self,
        slot: ElasticSlot,
        src_rank: int | None,
        dst_rank: int,
        parity_id: int | None,
        coeff: int,
        num_bytes: int,
        description: str,
    ) -> TransferOp:
        return TransferOp(
            src_rank=None if src_rank is None else int(src_rank),
            dst_rank=int(dst_rank),
            src_device="virtual_zero" if src_rank is None else self._rank_device(int(src_rank)),
            dst_device=self._rank_device(int(dst_rank)),
            slot_id=slot.slot_id,
            data_group_id=slot.data_group_id,
            relative_index=slot.relative_index,
            parity_id=parity_id,
            coeff=int(coeff) & 0xFF,
            num_bytes=num_bytes,
            is_virtual_zero=slot.is_virtual_zero,
            description=description,
        )


def make_planner(config: RacerConfig) -> SpareComputePlanner:
    return SpareComputePlanner(config)
